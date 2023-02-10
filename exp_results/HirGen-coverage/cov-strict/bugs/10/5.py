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
const_360 = relay.const([[[-6.863317,8.287552,-8.663866,-3.737082,3.198981,-6.395001,-3.189238,-0.161100,9.977456,9.556786,3.120777,-0.094019,6.883205,6.012859,-9.312407,4.499228]],[[-8.475973,-7.268421,2.370051,4.975543,-9.448960,-9.787854,6.720651,-9.866416,1.658718,3.011033,5.884394,1.417162,7.193188,-3.266036,5.244192,-7.774589]],[[-5.693719,8.044752,8.018721,-4.450285,-2.779963,-2.119602,2.347420,-6.345848,-4.124394,-6.921224,1.963958,8.049861,-5.167728,-6.049792,0.532716,8.569552]],[[1.522875,8.846539,1.123181,-2.471204,-1.958681,7.714933,8.113058,-1.650459,6.771697,1.715760,-4.107010,1.778073,-8.557806,4.323572,-7.548608,1.582999]],[[2.802167,-9.729326,-8.095870,-0.656270,4.842661,8.764632,6.548965,4.750806,-3.764451,1.510388,5.813229,1.036810,4.463589,1.953089,-1.714215,2.725340]],[[7.253106,-8.044602,1.622319,-7.716990,4.874949,0.001953,9.307836,5.702654,-7.649363,6.818166,-4.960002,2.959823,3.623859,5.266319,-7.694404,2.828249]],[[7.369426,8.048858,-4.983921,3.420617,6.201746,-1.016099,8.128537,-0.976635,-0.684053,0.213917,8.470014,-5.115903,7.590123,-8.367733,4.313200,4.049157]],[[7.959914,2.364419,-6.802375,7.483935,9.431610,6.221619,-4.620372,-5.527272,0.611468,6.338190,1.122714,0.743632,2.028343,-5.258347,7.383055,5.640291]]], dtype = "float64")#candidate|360|(8, 1, 16)|const|float64
var_361 = relay.var("var_361", dtype = "float64", shape = (8, 10, 16))#candidate|361|(8, 10, 16)|var|float64
bop_362 = relay.minimum(const_360.astype('float64'), var_361.astype('float64')) # shape=(8, 10, 16)
output = bop_362
output2 = bop_362
func_417 = relay.Function([var_361,], output)
mod['func_417'] = func_417
mod = relay.transform.InferType()(mod)
var_418 = relay.var("var_418", dtype = "float64", shape = (8, 10, 16))#candidate|418|(8, 10, 16)|var|float64
output = func_417(var_418)
func_419 = relay.Function([var_418], output)
mutated_mod['func_419'] = func_419
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1200 = relay.var("var_1200", dtype = "uint8", shape = ())#candidate|1200|()|var|uint8
var_1201 = relay.var("var_1201", dtype = "uint8", shape = (5, 9, 12))#candidate|1201|(5, 9, 12)|var|uint8
bop_1202 = relay.less_equal(var_1200.astype('bool'), var_1201.astype('bool')) # shape=(5, 9, 12)
output = bop_1202
output2 = bop_1202
func_1215 = relay.Function([var_1200,var_1201,], output)
mod['func_1215'] = func_1215
mod = relay.transform.InferType()(mod)
mutated_mod['func_1215'] = func_1215
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1215_call = mutated_mod.get_global_var('func_1215')
var_1217 = relay.var("var_1217", dtype = "uint8", shape = ())#candidate|1217|()|var|uint8
var_1218 = relay.var("var_1218", dtype = "uint8", shape = (5, 9, 12))#candidate|1218|(5, 9, 12)|var|uint8
call_1216 = func_1215_call(var_1217,var_1218,)
output = call_1216
func_1219 = relay.Function([var_1217,var_1218,], output)
mutated_mod['func_1219'] = func_1219
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1332 = relay.var("var_1332", dtype = "bool", shape = (10, 8, 9))#candidate|1332|(10, 8, 9)|var|bool
var_1333 = relay.var("var_1333", dtype = "bool", shape = (10, 8, 9))#candidate|1333|(10, 8, 9)|var|bool
bop_1334 = relay.logical_or(var_1332.astype('bool'), relay.reshape(var_1333.astype('bool'), relay.shape_of(var_1332))) # shape=(10, 8, 9)
func_417_call = mod.get_global_var('func_417')
func_419_call = mutated_mod.get_global_var('func_419')
var_1343 = relay.var("var_1343", dtype = "float64", shape = (1280,))#candidate|1343|(1280,)|var|float64
call_1342 = func_417_call(relay.reshape(var_1343.astype('float64'), [8, 10, 16]))
call_1344 = func_417_call(relay.reshape(var_1343.astype('float64'), [8, 10, 16]))
bop_1357 = relay.not_equal(var_1333.astype('bool'), relay.reshape(var_1332.astype('bool'), relay.shape_of(var_1333))) # shape=(10, 8, 9)
output = relay.Tuple([bop_1334,call_1342,var_1343,bop_1357,])
output2 = relay.Tuple([bop_1334,call_1344,var_1343,bop_1357,])
func_1364 = relay.Function([var_1332,var_1333,var_1343,], output)
mod['func_1364'] = func_1364
mod = relay.transform.InferType()(mod)
var_1365 = relay.var("var_1365", dtype = "bool", shape = (10, 8, 9))#candidate|1365|(10, 8, 9)|var|bool
var_1366 = relay.var("var_1366", dtype = "bool", shape = (10, 8, 9))#candidate|1366|(10, 8, 9)|var|bool
var_1367 = relay.var("var_1367", dtype = "float64", shape = (1280,))#candidate|1367|(1280,)|var|float64
output = func_1364(var_1365,var_1366,var_1367,)
func_1368 = relay.Function([var_1365,var_1366,var_1367,], output)
mutated_mod['func_1368'] = func_1368
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1470 = relay.var("var_1470", dtype = "float64", shape = (8, 5, 13))#candidate|1470|(8, 5, 13)|var|float64
uop_1471 = relay.asin(var_1470.astype('float64')) # shape=(8, 5, 13)
func_1364_call = mod.get_global_var('func_1364')
func_1368_call = mutated_mod.get_global_var('func_1368')
const_1474 = relay.const([True,False,False,True,True,False,True,False,True,True,False,False,False,True,False,True,True,True,False,True,False,True,False,False,True,True,True,False,True,True,True,False,True,True,True,True,False,True,True,True,False,True,True,False,True,False,True,False,True,True,False,False,True,False,False,False,True,True,False,True,True,True,True,False,True,True,True,True,False,False,False,False,False,True,False,True,True,False,True,True,True,True,True,False,True,True,False,True,True,True,False,False,False,True,False,True,False,False,False,False,False,False,False,False,False,True,True,True,True,True,False,False,False,True,True,False,True,True,True,False,False,True,False,True,False,False,False,True,False,True,True,True,True,False,True,True,True,True,True,False,False,True,False,False,True,True,False,False,True,True,False,True,False,False,False,True,False,True,False,True,False,True,False,True,True,True,True,False,False,False,False,False,True,True,True,False,False,True,False,True,True,False,True,True,False,True,False,True,False,False,False,False,False,False,False,True,True,True,False,False,True,False,False,False,True,True,True,False,True,True,True,False,False,False,True,False,False,False,True,False,False,True,True,False,True,True,False,True,False,False,True,True,False,True,True,False,True,False,False,False,False,True,False,False,False,True,False,False,True,False,False,False,True,True,False,True,False,False,False,False,False,True,True,True,False,True,True,True,True,True,True,True,False,True,True,False,True,False,False,False,False,True,False,True,False,True,False,True,True,False,True,False,True,True,True,False,False,False,True,True,True,True,True,True,False,False,False,True,False,False,False,False,True,False,False,False,True,False,True,True,True,False,True,False,True,False,False,True,False,False,True,True,True,False,False,True,False,False,True,True,True,True,True,False,True,True,False,True,False,True,False,True,False,True,True,True,True,False,True,True,False,False,False,True,False,False,False,False,True,True,True,False,False,True,False,False,False,True,True,False,False,True,True,False,False,True,False,True,True,True,False,True,True,False,False,True,False,False,True,True,False,True,True,False,False,False,False,False,True,True,True,True,False,False,False,False,True,False,True,False,True,True,True,False,True,True,True,False,False,False,True,False,True,True,False,True,True,True,False,False,False,True,True,True,True,True,True,True,True,True,True,False,False,True,False,False,False,True,False,False,False,True,False,True,False,True,True,True,False,True,True,False,False,True,True,True,False,True,False,False,False,False,False,False,True,True,False,True,False,False,False,False,False,False,False,False,True,True,False,True,False,True,False,False,False,True,True,True,False,False,True,False,False,True,True,True,False,True,False,True,False,False,True,False,True,True,False,False,False,False,False,False,False,False,False,False,True,False,True,False,False,False,False,False,False,True,True,False,True,True,True,True,False,True,True,True,False,True,True,True,False,True,True,False,True,False,False,True,False,False,True,False,False,True,False,False,True,True,True,False,False,False,True,False,True,False,True,False,True,True,True,True,False,False,True,False,False,False,True,False,False,False,False,False,True,False,True,False,True,False,False,False,False,True,False,False,True,False,False,True,True,True,False,True,True,False,True,False,False,False,False,False,False,True,False,False,True,True,False,True,True,True,True,False,False,True,False,False,True,False,True,False,True,True,False,False,True,True,False,True,False,True,True,False,False,False,False,True,True,True,False,True,False,True,True,False,True,True,False,False,True,True,True,False,True,True,False,False,False,True,False,False,False,True,True,False,True,True,True,True,False,False,False,False,True,True,True,False,False,True,True,True,True,False,True,False,True,True,False,True], dtype = "bool")#candidate|1474|(720,)|const|bool
const_1475 = relay.const([9.356556,2.069754,-7.804535,6.739082,-1.602509,6.428199,1.951542,3.056010,-0.257753,9.510092,6.869425,-0.313834,-6.257160,-4.175676,3.750718,7.214059,-7.117354,1.048517,-1.235428,5.638284,-8.765128,-1.453028,-4.153034,-0.421905,1.270442,-5.079493,-3.212691,-7.842771,-2.158865,8.566812,-9.049697,9.242943,6.669632,1.271097,1.025721,-1.897844,-7.206398,7.114411,9.519829,7.170865,0.576568,3.257397,-0.862710,2.476826,3.267671,9.871505,8.135975,-1.573500,3.236014,-7.205241,3.848448,-7.173080,6.040367,-1.610766,0.969327,4.077481,4.095214,-5.777068,-4.635258,2.132699,-7.972795,-9.609114,6.307038,-4.159440,-1.875337,9.194311,-8.212658,4.004280,-0.317311,-7.390200,4.969839,-6.708888,-5.415478,1.587146,-3.695482,-9.907149,5.242665,-6.607491,-5.163795,5.336217,2.804823,-3.485945,-1.213108,5.375050,2.540672,-0.905733,5.428479,-1.175039,7.557692,3.291152,-9.814445,7.594931,-8.738865,-1.065055,-8.701399,8.575210,-4.204704,5.106150,-8.218077,8.400205,-8.100334,-3.017265,3.661764,-4.697432,-4.391682,-5.468609,1.518957,-1.712342,-0.277665,-2.740795,-2.777695,-2.727481,3.911521,9.594840,1.721731,-3.413245,-2.319309,1.454666,7.955782,7.145514,5.474704,-2.989981,3.399068,-3.766645,0.405804,-8.567970,3.496528,-7.699262,3.341952,-1.894477,3.342449,-6.969282,8.669113,9.471353,7.082903,2.976462,-6.485608,-4.922674,-4.924195,9.247699,0.040540,-7.953486,0.737869,-5.079084,-8.490420,6.937976,0.783330,-3.201370,4.333123,4.677906,-6.519505,-4.202714,-0.927325,-8.819630,2.698581,-6.501970,4.113367,1.049768,7.184249,1.528869,-7.823789,3.561045,7.163225,7.541220,8.595887,0.888177,7.918831,5.678790,-7.420204,-9.377002,6.772126,3.667451,0.568704,7.193956,-6.984640,8.523409,9.272139,-9.644930,-4.735340,-2.483635,-8.858086,9.333270,8.210469,9.183654,-9.261148,1.831925,-2.613548,-8.824089,-2.879039,-1.101811,8.493763,8.852572,4.013092,1.015387,0.822080,7.611468,8.015903,8.609654,3.852670,2.260948,9.767764,-4.690783,9.954927,4.383589,-9.978059,-0.840263,2.100305,9.910444,2.184303,-4.226534,-7.351901,-6.921498,-9.183483,-3.478303,-7.186302,1.030864,-0.600635,4.265297,3.392340,3.113307,-6.204666,0.808577,-9.745743,-2.954562,0.682366,-2.787790,4.652039,8.653753,7.448316,7.837879,8.899843,0.125999,-1.398492,7.661527,7.729547,-5.596829,-6.720237,-8.887504,-4.813654,-1.574853,8.237458,7.761985,9.108807,3.332461,-1.928282,-7.030309,1.720475,5.921596,-5.712158,0.020989,-8.331160,-1.265290,3.093347,5.755544,3.199374,0.889692,-4.979948,3.761217,-7.341572,-6.760340,-8.357728,-0.433211,-1.491068,7.016161,2.390673,7.166557,-2.286666,-0.727426,-7.892709,2.791864,-7.822981,-5.872080,7.043398,-0.491864,-6.560666,-2.407602,-2.181298,7.570085,4.287505,0.345392,5.161517,-1.430124,-9.539437,-5.513401,-9.900711,9.615069,-5.524848,5.756715,8.935800,0.581818,6.127096,-6.689548,8.568960,-0.148180,2.784773,0.697324,-2.262073,-3.978400,-4.353476,8.185424,9.730110,8.471235,3.067225,6.777785,3.850341,-8.889748,2.400025,-2.544400,-1.836903,7.927020,-2.423780,7.043695,-9.347739,-5.930619,-9.033222,5.949788,-6.732406,8.998349,1.351796,-9.806225,-4.229357,9.914115,-7.888753,-0.223408,3.300746,-3.406557,-1.267245,-5.782745,6.672904,2.270816,2.432750,1.872253,-7.235726,8.640684,-8.390676,-2.672336,-1.841090,-4.057465,1.852593,-2.956231,-0.546798,-9.462172,-7.016000,4.443796,3.745049,8.966282,9.774248,-5.166004,6.520607,9.116349,-6.143237,-7.187520,-2.714156,2.512003,5.916590,-4.310231,-9.186417,-9.965698,5.888196,-3.271613,1.240170,0.369865,8.580996,-5.907353,2.473049,-9.385876,7.651044,6.463630,7.819067,4.958850,-4.241088,-5.173587,-9.819556,0.645944,-7.174372,3.321204,1.062327,-4.606677,-5.832574,6.649408,-8.352049,1.435306,6.970862,5.855981,5.737604,7.789549,-3.007750,-3.448178,5.627296,0.385766,-3.388077,5.245886,4.268301,-1.674516,-0.711992,-8.963656,9.848901,-6.872826,-4.681972,2.634282,-0.696368,7.946536,-0.131708,2.035575,2.022774,7.616147,8.945251,9.529947,-8.421742,7.943616,2.949814,0.013688,9.693845,5.447757,-9.181726,-7.931070,-1.506047,8.300774,3.811070,-4.183295,5.664106,-8.931805,-2.610922,-0.822538,3.631230,3.521309,2.347995,0.940432,-0.108206,-8.137750,6.259986,3.428376,-5.109939,-5.378686,5.024814,-2.766578,-2.907305,-0.006478,-1.970906,5.933449,-8.864975,-2.836753,-6.746594,-3.127607,2.488084,1.694364,0.685257,-4.524690,4.633458,-2.908162,4.563250,-4.798521,3.722785,-7.516467,3.189315,-2.296599,-1.141969,7.236081,-2.432475,1.605420,-5.824874,7.493072,9.729627,4.008488,-0.285477,4.586976,-4.034747,-6.263466,-0.385038,3.353734,-4.184363,6.431404,-4.417951,9.525108,9.268669,-9.172562,6.905878,5.812802,4.130890,-2.474499,-4.843187,6.288164,4.022009,2.438366,-1.055495,9.127233,-1.964419,5.248051,4.147770,4.355533,8.736798,-6.070097,8.076610,9.863897,-6.656702,0.789047,-3.440456,9.045673,-4.809992,9.251958,0.355290,0.516206,-6.698199,-1.806862,-5.479994,-5.863210,-8.742964,-3.646794,-1.312512,-3.473536,8.297847,-2.243624,1.914574,2.131768,-6.988321,-4.805488,1.093354,2.709751,5.899027,4.548839,0.649078,8.400335,-0.576914,-6.967183,-6.395513,1.773112,-4.249358,2.397304,6.614444,9.236463,7.566546,2.115410,8.119970,-0.810019,6.115239,-4.600391,4.671596,-8.654311,-6.412657,3.835217,4.483451,-2.438398,-7.226156,1.531874,7.108305,9.208171,8.392484,3.956017,8.937240,1.549804,-9.011603,9.223059,3.536693,5.601822,-7.327091,-5.691207,-2.358128,8.800201,-7.510634,-4.508268,7.306533,8.650245,4.318254,-9.146640,-2.930965,3.718121,6.216487,7.713814,-6.761314,-0.818193,0.095078,-5.621069,-5.044475,-2.819063,3.658303,-9.032768,-1.144703,5.334537,-3.560221,-4.333285,9.038286,5.692438,-2.718521,4.911099,-1.785864,-9.368580,4.331194,6.103335,-2.723650,-2.292361,-9.894278,6.593549,-1.672530,-7.576344,-9.605118,-3.598081,-4.689671,-8.309217,-2.156453,7.463889,-1.800317,-8.185252,1.396121,4.438742,5.484763,2.997118,-2.995316,-1.734558,5.598692,7.583877,0.602172,7.263207,-8.033056,-3.585741,7.732145,6.039705,8.007885,0.796454,3.976099,0.989366,1.089074,-3.707936,1.317215,-8.355554,-9.689565,-4.427368,7.082181,9.172955,8.747154,-9.695071,-2.919032,-7.606582,2.385877,-9.336899,-8.183649,7.086606,-7.684000,0.342504,-1.248009,-7.313860,-9.351228,-0.356815,-0.821001,8.791417,-5.500695,-2.023284,9.937389,-3.184889,1.327210,5.731882,-6.922923,1.989118,-4.620643,-8.859511,-8.507254,5.646428,8.607048,1.913942,-1.099140,-2.785391,-2.855259,9.995045,9.791489,7.002358,-3.068000,5.325867,9.308036,2.198600,6.247894,3.707624,-1.633257,-2.461899,-4.283984,5.360046,-3.878908,7.905020,-5.931759,-4.791485,4.125183,-2.296943,-7.345948,-5.115009,-8.494339,-8.982824,-5.934490,-1.632867,-9.541797,-5.423061,-9.513294,9.875116,4.761434,7.056111,8.460596,-5.786573,7.657010,-9.715850,-6.965119,-8.925761,-5.118467,-8.713754,-0.012321,3.774241,-1.400594,0.749796,4.037848,9.982979,-8.080316,7.052992,1.816312,-1.124245,-3.621633,-1.712012,7.947734,-7.917114,2.214399,-4.794942,0.168487,9.388191,-9.200664,-9.033695,1.471023,-7.288207,-0.062093,-6.999190,9.768293,9.884239,1.550213,-4.843532,-4.508025,7.210196,-4.509298,5.500110,6.214477,8.886712,1.231061,-9.355547,5.966036,-3.443595,-9.163027,7.677421,3.433902,8.291450,0.897659,-5.456296,-7.085911,0.363774,4.435424,-0.780644,4.436333,-8.790201,0.049902,-6.664737,-5.351939,2.768138,3.915369,-8.824913,6.498524,5.583513,-0.564407,-4.719720,7.020593,9.186704,-1.321828,-9.415553,9.661212,-9.244045,5.052584,9.597909,-3.697234,-3.281516,-5.598849,-8.768885,-9.214075,-5.455876,5.905596,-4.651336,-8.508500,-8.127652,4.355291,6.715235,6.820841,-1.604011,-0.203673,-5.544652,-0.058364,-5.069333,-8.674927,-4.892434,-8.525330,8.759708,1.289357,9.853077,4.869820,-6.913950,-5.299391,1.256402,7.063945,-4.941121,0.773128,-3.564929,6.524561,-4.250834,-6.636112,-0.218947,0.430623,-0.876800,8.124878,2.494119,4.608215,-3.943537,4.702795,-5.793184,0.312637,-4.459384,8.324584,-4.715786,5.842431,-9.681830,7.249560,-1.681527,-3.790295,6.977639,-2.836202,-7.569398,-0.408122,3.888087,-2.293784,-6.197646,-4.743252,9.787586,-7.770652,-9.641607,-4.296363,1.313132,-2.508097,9.931951,-6.895528,-4.361051,7.670808,9.152295,3.877231,0.639377,-3.616285,4.209415,-7.049085,-5.459311,-3.832603,-8.054552,-1.967186,-7.889732,7.730832,1.085782,-0.572297,9.876522,7.803178,3.520255,0.998064,-6.468370,5.685978,0.082190,-3.377753,-7.920413,-9.682640,-7.169545,-4.203826,0.739576,-9.657863,6.786737,3.992443,-9.667864,4.299225,5.193823,-1.464365,-8.291502,-2.155853,8.706471,-4.619974,0.061925,-5.671026,7.834936,0.146040,-1.201611,3.187568,-6.140029,-0.204136,7.582189,4.743640,5.721008,0.246539,6.114988,1.294210,-7.248046,-2.137259,-3.275906,3.031714,6.300312,4.200541,-5.783540,6.426181,2.178305,9.720386,-5.436865,2.972503,-5.095451,1.476661,-5.248067,-0.063599,-7.695568,6.233437,9.467090,-0.099397,1.480292,3.931069,2.460370,-0.700171,-2.665236,1.499431,-4.756878,-0.199833,4.594253,-7.170298,-9.073491,3.228768,1.233974,-9.266784,6.586688,4.845540,-8.527042,0.836125,-1.902852,-3.323290,4.344226,1.629747,-4.449068,2.409065,7.452245,-9.165738,-0.037599,7.436374,-8.287974,6.757231,-6.301783,-2.477576,6.962901,-4.988892,5.729909,-6.553660,7.064281,1.386257,-6.091083,0.146292,-7.900338,6.673184,8.960189,9.265290,-2.924588,-3.782577,3.436569,1.222215,-3.465370,-9.666219,-6.317384,-6.090092,-6.231789,-3.779411,-9.922686,1.670468,1.499947,-4.021007,2.711057,-7.133435,-8.974961,-6.706055,1.817067,-7.870281,-5.173658,1.485865,7.217532,-9.391344,2.051576,-8.283831,-8.780638,2.016368,7.644861,1.650668,4.580049,-2.141553,-8.140883,5.169724,7.174302,0.091922,2.939249,-2.371547,-0.704173,-4.168791,2.604064,-5.708670,-2.755809,1.231306,2.674731,-3.537594,-8.775028,-0.493618,-0.083512,-3.084995,9.071028,9.613980,8.600612,4.677432,-7.513336,3.459127,3.135165,3.123955,8.188595,8.387805,3.899030,5.026098,-6.853720,-3.243469,-3.164805,-9.813666,-4.229989,-8.524729,-4.753072,-9.737722,4.266496,-6.999810,9.301081,5.930820,4.671991,2.332539,-4.526956,-8.523237,-0.219358,4.152642,-8.059964,-6.470850,8.810856,-1.701855,-9.131875,-0.367219,-5.101711,-5.136142,3.773397,0.230850,-6.019905,6.308778,6.848435,-4.825446,7.119510,-2.580191,9.196155,-1.202190,1.820035,-0.099588,9.842339,-1.586008,-1.388787,9.381565,-4.494449,-7.628210,6.453448,0.310039,9.445452,-0.269443,-8.198163,-9.877991,3.764296,-7.558742,4.949842,7.271415,-7.989994,7.400329,-6.902972,-9.304764,6.783278,-9.188004,-7.657865,2.950411,0.102120,-3.919990,2.822409,1.908865,-8.041751,3.276577,5.267616,-2.035443,-5.323451,-2.195992,4.389457,2.038875,9.674806,9.529114,-3.487882,1.024293,6.544338,0.859655,7.440334,8.490334,-2.449960,3.388047,7.835528,1.636322,9.143129,3.768191,-3.286993,1.577227,-0.658303,-4.690492,-3.335115,-8.022004,1.766871,-5.438070,-2.565535,-5.631898,-7.960411,-7.638677,-3.288999,-2.840335,0.904221,-8.917394,-7.918997,2.561859,-7.438393,6.888572,-7.917007,-3.572448,3.546290,-6.858809,4.475732,7.931810,-2.924914,0.120680,1.936894,5.990973,9.084239,0.515090,7.343434,-9.859405,-7.001128,-2.832234,1.557209,-5.611697,8.323855,4.125771,-5.756863,7.125644,-1.689040,7.638313,-9.075845,9.897742,-1.602046,5.076868,8.485529,9.213291,-7.546019,-5.702801,3.066865,-1.297664,4.445084,9.774470,-2.496359,5.275059,4.205102,9.726783,3.353589,-1.516208,1.911661,7.701993,6.513163,-5.876706,2.394537,-0.652756,6.804382,5.794023,-2.096687,-5.486103,4.517124,2.547121,-3.424325,-3.630264,7.158680,2.969482,9.014005,-1.030559,-4.711130,-4.118800,1.039321,-2.465165,1.236442,-8.666700,-5.115410,8.698229,-7.660550,-1.645736,4.328327,-3.638863,9.791742,0.190576,-1.031750,-2.751443,-6.189825,-2.306208,4.486305,-1.860202,4.174765,-7.663820,-0.703499,3.651225,-6.128720,5.226591,4.617194,1.790574,-2.884639,4.204582,-6.076898,3.611667,-1.290248,-9.045546,-3.761396,-1.022376,1.571837,1.846874,-7.338070,-1.516057,-4.413212,-5.753491,9.789348,9.112815,-6.075715,8.278808,-5.393018,7.137869,-1.936913,-1.950042,8.959589,-0.813722,1.687073,-7.902048,3.759587,-1.874977,-8.966695,6.843094,-3.926243,1.089043,2.085287,2.383729,6.567287,-6.605750,2.701690,-1.068413,2.139628,0.138063,-7.969162,7.110508,8.777059,-7.105176,-7.655964,8.673808,-8.502460,-0.229422,3.060089,3.247000,-0.817949,7.580256,-2.206061,-7.427712,-8.887023,-7.600329,6.561806,-2.129734,3.780429,2.989944,-7.383240,0.326281,7.646004,-4.735302,9.581496,-2.096333,6.702563], dtype = "float64")#candidate|1475|(1280,)|const|float64
call_1473 = relay.TupleGetItem(func_1364_call(relay.reshape(const_1474.astype('bool'), [10, 8, 9]), relay.reshape(const_1474.astype('bool'), [10, 8, 9]), relay.reshape(const_1475.astype('float64'), [1280,]), ), 0)
call_1476 = relay.TupleGetItem(func_1368_call(relay.reshape(const_1474.astype('bool'), [10, 8, 9]), relay.reshape(const_1474.astype('bool'), [10, 8, 9]), relay.reshape(const_1475.astype('float64'), [1280,]), ), 0)
func_1364_call = mod.get_global_var('func_1364')
func_1368_call = mutated_mod.get_global_var('func_1368')
call_1482 = relay.TupleGetItem(func_1364_call(relay.reshape(call_1473.astype('bool'), [10, 8, 9]), relay.reshape(const_1474.astype('bool'), [10, 8, 9]), relay.reshape(const_1475.astype('float64'), [1280,]), ), 2)
call_1483 = relay.TupleGetItem(func_1368_call(relay.reshape(call_1473.astype('bool'), [10, 8, 9]), relay.reshape(const_1474.astype('bool'), [10, 8, 9]), relay.reshape(const_1475.astype('float64'), [1280,]), ), 2)
output = relay.Tuple([uop_1471,call_1473,const_1474,const_1475,call_1482,])
output2 = relay.Tuple([uop_1471,call_1476,const_1474,const_1475,call_1483,])
func_1497 = relay.Function([var_1470,], output)
mod['func_1497'] = func_1497
mod = relay.transform.InferType()(mod)
var_1498 = relay.var("var_1498", dtype = "float64", shape = (8, 5, 13))#candidate|1498|(8, 5, 13)|var|float64
output = func_1497(var_1498)
func_1499 = relay.Function([var_1498], output)
mutated_mod['func_1499'] = func_1499
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1707 = relay.var("var_1707", dtype = "int8", shape = (9, 9, 9))#candidate|1707|(9, 9, 9)|var|int8
const_1708 = relay.const([[[-8,-4,-7,-8,-4,3,-4,4,-8],[-2,7,1,-10,-10,-2,2,2,-7],[-6,-1,7,-4,10,-10,10,-1,2],[7,-10,8,9,-3,7,-2,-4,-3],[-7,-10,2,-7,2,1,1,-7,-4],[2,4,-6,5,3,-1,8,3,6],[10,10,-3,-7,7,-5,6,4,-3],[2,-10,-7,1,-10,-5,-10,10,-6],[-10,-7,-6,8,2,-3,-4,-4,2]],[[-7,7,9,-4,9,-8,-8,10,5],[-8,10,-7,-4,2,5,-9,8,2],[2,-6,-1,-9,-5,6,7,-3,5],[-1,-10,9,6,-10,6,10,9,10],[-2,-10,-8,-7,2,1,9,-8,-3],[-6,5,-3,-9,8,1,-1,4,10],[-10,6,-4,-1,7,3,6,-10,7],[-1,-8,-6,6,-10,9,-2,-6,2],[3,4,-3,-1,6,8,-7,-10,8]],[[4,3,4,-5,-2,-7,10,10,-2],[-10,-1,-6,-10,10,8,7,-1,5],[4,-4,-10,5,-10,3,-7,7,-9],[-7,-8,10,4,-9,10,2,5,-2],[-6,-4,-1,2,-9,-6,-4,9,9],[-7,9,-7,1,2,-7,10,1,-2],[10,-10,3,-7,-10,-8,-6,-1,-2],[-3,-10,-8,7,-10,1,-3,-8,-9],[-1,-9,-3,-10,-3,-5,3,-9,-10]],[[10,6,-10,-6,1,7,6,-6,-9],[-3,1,3,-2,-3,9,-9,-5,-1],[-5,7,10,5,2,-2,-2,3,2],[1,-2,6,-2,-10,-1,1,7,-3],[1,-2,1,7,10,9,6,5,-6],[9,5,4,-4,3,2,8,-6,-10],[6,-2,-7,-8,3,-1,2,9,-3],[5,-8,9,-6,7,9,1,-5,4],[8,-5,3,1,7,-3,3,-6,-6]],[[-10,1,-6,-5,8,2,3,3,-10],[-3,-2,-3,1,-4,-6,3,5,-5],[6,10,9,-7,-1,-10,7,-3,6],[-5,-1,-7,1,3,5,9,-8,-7],[-10,-1,5,3,7,-2,-2,-3,9],[-8,-1,8,-10,9,-7,-7,3,9],[9,1,-8,10,-3,10,2,-2,-2],[-9,9,-10,-4,-5,-3,6,-6,10],[3,3,3,-8,5,8,-2,10,4]],[[-10,-5,7,1,1,-4,3,-3,3],[-10,-2,9,-2,8,-2,2,-1,9],[6,-3,9,-10,-5,9,-10,7,5],[6,8,-1,-6,-4,-9,-7,5,4],[7,8,-8,10,-9,-3,-6,4,-1],[-6,-8,-4,-5,2,10,8,9,-1],[-5,6,-1,-8,8,-1,-9,-5,-9],[-3,-9,6,10,-9,10,-7,-3,6],[-4,1,8,-10,-8,3,-6,-2,3]],[[6,-2,4,7,4,6,3,8,-1],[-5,5,-8,3,-7,8,-4,5,9],[7,-1,-5,-3,-9,10,7,-1,-1],[1,2,8,7,-2,8,-2,-9,-6],[-3,1,-9,-2,-8,-9,3,-4,9],[2,-4,10,8,-8,6,1,2,1],[-3,-10,-7,2,10,-2,-8,6,-1],[-7,-8,-7,-7,-6,-10,-4,10,-6],[4,5,-3,-4,-3,4,-7,-4,4]],[[-3,-8,3,6,1,-9,2,-6,-9],[6,-10,4,-3,-5,-9,-8,3,9],[-6,-1,-10,-5,-6,-1,-9,-8,8],[-5,6,7,10,-8,-1,-6,2,5],[10,6,10,10,-6,8,-2,-4,2],[6,9,6,-8,9,10,4,4,5],[-5,1,3,5,4,-6,3,10,-3],[-3,9,-3,-6,10,-6,3,-7,-5],[6,8,1,6,-5,7,-3,2,-4]],[[2,3,1,-4,-1,1,-3,-7,-8],[-6,-2,3,2,-2,9,-1,-8,-10],[4,-3,6,10,-4,-4,-5,9,3],[9,-1,2,-6,10,7,-1,7,-5],[-2,4,-4,-8,-1,8,-10,-3,-5],[2,8,-5,3,6,-4,3,-8,9],[7,2,-4,-4,-5,-7,2,4,-1],[-6,9,-2,-5,-6,-4,-6,2,-1],[-8,5,1,-6,3,-2,-2,-9,-4]]], dtype = "int8")#candidate|1708|(9, 9, 9)|const|int8
bop_1709 = relay.greater(var_1707.astype('bool'), relay.reshape(const_1708.astype('bool'), relay.shape_of(var_1707))) # shape=(9, 9, 9)
bop_1715 = relay.equal(const_1708.astype('bool'), relay.reshape(var_1707.astype('bool'), relay.shape_of(const_1708))) # shape=(9, 9, 9)
uop_1741 = relay.atan(var_1707.astype('float64')) # shape=(9, 9, 9)
output = relay.Tuple([bop_1709,bop_1715,uop_1741,])
output2 = relay.Tuple([bop_1709,bop_1715,uop_1741,])
func_1747 = relay.Function([var_1707,], output)
mod['func_1747'] = func_1747
mod = relay.transform.InferType()(mod)
var_1748 = relay.var("var_1748", dtype = "int8", shape = (9, 9, 9))#candidate|1748|(9, 9, 9)|var|int8
output = func_1747(var_1748)
func_1749 = relay.Function([var_1748], output)
mutated_mod['func_1749'] = func_1749
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1934 = relay.var("var_1934", dtype = "float32", shape = (6, 7, 1))#candidate|1934|(6, 7, 1)|var|float32
uop_1935 = relay.sin(var_1934.astype('float32')) # shape=(6, 7, 1)
output = relay.Tuple([uop_1935,])
output2 = relay.Tuple([uop_1935,])
func_1937 = relay.Function([var_1934,], output)
mod['func_1937'] = func_1937
mod = relay.transform.InferType()(mod)
var_1938 = relay.var("var_1938", dtype = "float32", shape = (6, 7, 1))#candidate|1938|(6, 7, 1)|var|float32
output = func_1937(var_1938)
func_1939 = relay.Function([var_1938], output)
mutated_mod['func_1939'] = func_1939
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1970 = relay.var("var_1970", dtype = "float32", shape = (11, 12, 15))#candidate|1970|(11, 12, 15)|var|float32
uop_1971 = relay.exp(var_1970.astype('float32')) # shape=(11, 12, 15)
func_1937_call = mod.get_global_var('func_1937')
func_1939_call = mutated_mod.get_global_var('func_1939')
const_1978 = relay.const([-6.563536,-5.485646,-0.652957,-6.473589,-7.940274,5.315644,-9.635703,-7.002357,-3.454026,8.422679,-5.811083,1.151619,4.587087,-7.991746,3.840574,4.653181,-0.766952,-2.130826,6.770656,-8.059539,-9.075668,-0.796209,6.587789,7.109744,-6.188972,-9.904743,6.614211,9.301222,1.647572,-3.941911,0.236032,-7.928276,9.731745,-8.587267,7.961008,2.177489,7.344060,-1.481051,-4.064620,-7.962102,-4.202356,3.167629], dtype = "float32")#candidate|1978|(42,)|const|float32
call_1977 = relay.TupleGetItem(func_1937_call(relay.reshape(const_1978.astype('float32'), [6, 7, 1])), 0)
call_1979 = relay.TupleGetItem(func_1939_call(relay.reshape(const_1978.astype('float32'), [6, 7, 1])), 0)
func_1937_call = mod.get_global_var('func_1937')
func_1939_call = mutated_mod.get_global_var('func_1939')
call_2001 = relay.TupleGetItem(func_1937_call(relay.reshape(const_1978.astype('float32'), [6, 7, 1])), 0)
call_2002 = relay.TupleGetItem(func_1939_call(relay.reshape(const_1978.astype('float32'), [6, 7, 1])), 0)
output = relay.Tuple([uop_1971,call_1977,const_1978,call_2001,])
output2 = relay.Tuple([uop_1971,call_1979,const_1978,call_2002,])
func_2016 = relay.Function([var_1970,], output)
mod['func_2016'] = func_2016
mod = relay.transform.InferType()(mod)
mutated_mod['func_2016'] = func_2016
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2017 = relay.var("var_2017", dtype = "float32", shape = (11, 12, 15))#candidate|2017|(11, 12, 15)|var|float32
func_2016_call = mutated_mod.get_global_var('func_2016')
call_2018 = func_2016_call(var_2017)
output = call_2018
func_2019 = relay.Function([var_2017], output)
mutated_mod['func_2019'] = func_2019
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2947 = relay.var("var_2947", dtype = "float64", shape = (11, 6, 10))#candidate|2947|(11, 6, 10)|var|float64
var_2948 = relay.var("var_2948", dtype = "float64", shape = (11, 6, 10))#candidate|2948|(11, 6, 10)|var|float64
bop_2949 = relay.floor_divide(var_2947.astype('float64'), relay.reshape(var_2948.astype('float64'), relay.shape_of(var_2947))) # shape=(11, 6, 10)
func_1215_call = mod.get_global_var('func_1215')
func_1219_call = mutated_mod.get_global_var('func_1219')
var_2987 = relay.var("var_2987", dtype = "uint8", shape = ())#candidate|2987|()|var|uint8
const_2988 = relay.const([[1,-2,-7,9,-10,-1,6,3,-3,-7,-3,-4,-8,6,8,-4,-6,-10,-1,10,-8,6,-1,7,7,7,-5,-5,-3,-4,-5,-1,-3,3,-10,5,-5,1,1,-2,-1,10,-3,8,-8,-6,-9,5,8,9,-4,-5,7,-7,-5,9,-1,2,8,9],[-2,-10,-3,-6,6,8,-10,-10,-2,-9,7,5,-9,-1,10,-4,8,5,10,2,-9,-7,-7,9,-4,-6,-8,-1,1,5,-6,4,-2,3,2,-9,8,-3,-3,-4,-1,-1,3,-6,-6,6,8,3,-8,-2,-6,3,-7,10,-9,8,-7,8,1,-6],[5,-8,-9,4,9,-8,-9,8,-9,7,-5,7,5,1,9,4,7,-8,-3,6,10,-9,-6,8,-7,1,-1,-1,-4,-4,-10,-9,-10,-5,-1,9,4,-4,-4,6,-7,-3,-5,-7,-7,-4,-1,5,-1,-8,-3,8,2,-8,2,8,-10,8,-5,8],[7,-10,6,2,10,-2,5,-10,9,5,-1,1,-4,-9,-5,3,-9,-2,10,2,4,-3,-1,-7,-9,2,-5,-3,3,10,-1,3,7,4,-9,-9,8,5,-3,2,5,3,-8,-1,-10,-10,-5,10,-8,1,-1,-2,10,-10,10,-6,7,-6,-10,5],[-7,1,-8,6,-5,3,5,7,-5,-2,-1,3,-2,3,6,-1,5,-7,-5,8,9,-8,-1,4,3,4,-4,10,-5,-5,-2,-4,-2,-1,5,9,-3,10,3,3,-8,8,3,8,-4,7,-1,3,5,10,-5,3,9,3,5,-8,2,-7,-7,-3],[9,-4,-4,-4,3,8,-7,6,-4,6,7,-4,-1,6,-5,-8,2,5,-1,2,4,-3,-10,-5,-10,-4,8,-7,-1,6,5,-10,-5,10,3,3,-3,8,-5,-10,7,-1,-3,3,-5,-10,4,5,5,7,-1,-5,5,10,10,7,5,2,8,2],[6,-4,-1,8,-6,4,-7,2,6,-4,-3,4,8,-7,-1,10,2,6,4,-1,-10,6,6,1,-3,6,3,5,6,-10,1,-2,5,-8,3,9,-4,1,4,-10,7,2,-8,3,-9,8,-4,-6,-10,10,-4,9,4,8,1,5,-7,-10,-6,-6],[-8,6,7,-5,-4,2,-1,-8,10,3,5,3,-4,7,-8,4,9,9,10,5,-6,4,8,2,-9,3,-2,-7,9,-6,-7,2,10,2,-3,6,3,-9,-3,3,6,7,2,4,-8,-6,-3,-8,10,-3,-6,6,-8,-4,-6,4,-3,-1,6,6],[-8,-8,-3,-3,8,-9,4,4,8,4,7,9,-8,-6,-8,8,-9,6,-3,-6,-7,-5,-8,-3,-9,4,5,-5,-4,5,5,7,-10,-7,-5,3,-10,3,9,6,-2,-3,6,10,-1,1,10,2,-4,-2,4,9,6,-9,7,-1,10,6,4,-5]], dtype = "uint8")#candidate|2988|(9, 60)|const|uint8
call_2986 = func_1215_call(relay.reshape(var_2987.astype('uint8'), []), relay.reshape(const_2988.astype('uint8'), [5, 9, 12]), )
call_2989 = func_1215_call(relay.reshape(var_2987.astype('uint8'), []), relay.reshape(const_2988.astype('uint8'), [5, 9, 12]), )
output = relay.Tuple([bop_2949,call_2986,var_2987,const_2988,])
output2 = relay.Tuple([bop_2949,call_2989,var_2987,const_2988,])
func_2996 = relay.Function([var_2947,var_2948,var_2987,], output)
mod['func_2996'] = func_2996
mod = relay.transform.InferType()(mod)
var_2997 = relay.var("var_2997", dtype = "float64", shape = (11, 6, 10))#candidate|2997|(11, 6, 10)|var|float64
var_2998 = relay.var("var_2998", dtype = "float64", shape = (11, 6, 10))#candidate|2998|(11, 6, 10)|var|float64
var_2999 = relay.var("var_2999", dtype = "uint8", shape = ())#candidate|2999|()|var|uint8
output = func_2996(var_2997,var_2998,var_2999,)
func_3000 = relay.Function([var_2997,var_2998,var_2999,], output)
mutated_mod['func_3000'] = func_3000
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3134 = relay.var("var_3134", dtype = "float64", shape = (5, 15, 1))#candidate|3134|(5, 15, 1)|var|float64
var_3135 = relay.var("var_3135", dtype = "float64", shape = (5, 15, 7))#candidate|3135|(5, 15, 7)|var|float64
bop_3136 = relay.add(var_3134.astype('float64'), var_3135.astype('float64')) # shape=(5, 15, 7)
output = relay.Tuple([bop_3136,])
output2 = relay.Tuple([bop_3136,])
func_3139 = relay.Function([var_3134,var_3135,], output)
mod['func_3139'] = func_3139
mod = relay.transform.InferType()(mod)
mutated_mod['func_3139'] = func_3139
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3139_call = mutated_mod.get_global_var('func_3139')
var_3141 = relay.var("var_3141", dtype = "float64", shape = (5, 15, 1))#candidate|3141|(5, 15, 1)|var|float64
var_3142 = relay.var("var_3142", dtype = "float64", shape = (5, 15, 7))#candidate|3142|(5, 15, 7)|var|float64
call_3140 = func_3139_call(var_3141,var_3142,)
output = call_3140
func_3143 = relay.Function([var_3141,var_3142,], output)
mutated_mod['func_3143'] = func_3143
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3256 = relay.var("var_3256", dtype = "uint16", shape = (6, 8, 7))#candidate|3256|(6, 8, 7)|var|uint16
var_3257 = relay.var("var_3257", dtype = "uint16", shape = (6, 8, 7))#candidate|3257|(6, 8, 7)|var|uint16
bop_3258 = relay.less_equal(var_3256.astype('bool'), relay.reshape(var_3257.astype('bool'), relay.shape_of(var_3256))) # shape=(6, 8, 7)
output = bop_3258
output2 = bop_3258
func_3278 = relay.Function([var_3256,var_3257,], output)
mod['func_3278'] = func_3278
mod = relay.transform.InferType()(mod)
mutated_mod['func_3278'] = func_3278
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3278_call = mutated_mod.get_global_var('func_3278')
var_3280 = relay.var("var_3280", dtype = "uint16", shape = (6, 8, 7))#candidate|3280|(6, 8, 7)|var|uint16
var_3281 = relay.var("var_3281", dtype = "uint16", shape = (6, 8, 7))#candidate|3281|(6, 8, 7)|var|uint16
call_3279 = func_3278_call(var_3280,var_3281,)
output = call_3279
func_3282 = relay.Function([var_3280,var_3281,], output)
mutated_mod['func_3282'] = func_3282
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3356 = relay.var("var_3356", dtype = "float64", shape = (2, 16, 2))#candidate|3356|(2, 16, 2)|var|float64
uop_3357 = relay.log(var_3356.astype('float64')) # shape=(2, 16, 2)
output = relay.Tuple([uop_3357,])
output2 = relay.Tuple([uop_3357,])
func_3359 = relay.Function([var_3356,], output)
mod['func_3359'] = func_3359
mod = relay.transform.InferType()(mod)
var_3360 = relay.var("var_3360", dtype = "float64", shape = (2, 16, 2))#candidate|3360|(2, 16, 2)|var|float64
output = func_3359(var_3360)
func_3361 = relay.Function([var_3360], output)
mutated_mod['func_3361'] = func_3361
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4098 = relay.var("var_4098", dtype = "float64", shape = (12, 3, 1))#candidate|4098|(12, 3, 1)|var|float64
var_4099 = relay.var("var_4099", dtype = "float64", shape = (12, 3, 1))#candidate|4099|(12, 3, 1)|var|float64
bop_4100 = relay.floor_mod(var_4098.astype('float64'), relay.reshape(var_4099.astype('float64'), relay.shape_of(var_4098))) # shape=(12, 3, 1)
output = bop_4100
output2 = bop_4100
func_4105 = relay.Function([var_4098,var_4099,], output)
mod['func_4105'] = func_4105
mod = relay.transform.InferType()(mod)
mutated_mod['func_4105'] = func_4105
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4105_call = mutated_mod.get_global_var('func_4105')
var_4107 = relay.var("var_4107", dtype = "float64", shape = (12, 3, 1))#candidate|4107|(12, 3, 1)|var|float64
var_4108 = relay.var("var_4108", dtype = "float64", shape = (12, 3, 1))#candidate|4108|(12, 3, 1)|var|float64
call_4106 = func_4105_call(var_4107,var_4108,)
output = call_4106
func_4109 = relay.Function([var_4107,var_4108,], output)
mutated_mod['func_4109'] = func_4109
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4127 = relay.var("var_4127", dtype = "float32", shape = (8, 11, 9))#candidate|4127|(8, 11, 9)|var|float32
uop_4128 = relay.sinh(var_4127.astype('float32')) # shape=(8, 11, 9)
func_417_call = mod.get_global_var('func_417')
func_419_call = mutated_mod.get_global_var('func_419')
var_4133 = relay.var("var_4133", dtype = "float64", shape = (1280,))#candidate|4133|(1280,)|var|float64
call_4132 = func_417_call(relay.reshape(var_4133.astype('float64'), [8, 10, 16]))
call_4134 = func_417_call(relay.reshape(var_4133.astype('float64'), [8, 10, 16]))
func_417_call = mod.get_global_var('func_417')
func_419_call = mutated_mod.get_global_var('func_419')
call_4143 = func_417_call(relay.reshape(call_4132.astype('float64'), [8, 10, 16]))
call_4144 = func_417_call(relay.reshape(call_4132.astype('float64'), [8, 10, 16]))
uop_4156 = relay.sigmoid(uop_4128.astype('float32')) # shape=(8, 11, 9)
func_3278_call = mod.get_global_var('func_3278')
func_3282_call = mutated_mod.get_global_var('func_3282')
var_4175 = relay.var("var_4175", dtype = "uint16", shape = (336,))#candidate|4175|(336,)|var|uint16
call_4174 = func_3278_call(relay.reshape(var_4175.astype('uint16'), [6, 8, 7]), relay.reshape(var_4175.astype('uint16'), [6, 8, 7]), )
call_4176 = func_3278_call(relay.reshape(var_4175.astype('uint16'), [6, 8, 7]), relay.reshape(var_4175.astype('uint16'), [6, 8, 7]), )
output = relay.Tuple([call_4132,var_4133,call_4143,uop_4156,call_4174,var_4175,])
output2 = relay.Tuple([call_4134,var_4133,call_4144,uop_4156,call_4176,var_4175,])
func_4185 = relay.Function([var_4127,var_4133,var_4175,], output)
mod['func_4185'] = func_4185
mod = relay.transform.InferType()(mod)
mutated_mod['func_4185'] = func_4185
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4185_call = mutated_mod.get_global_var('func_4185')
var_4187 = relay.var("var_4187", dtype = "float32", shape = (8, 11, 9))#candidate|4187|(8, 11, 9)|var|float32
var_4188 = relay.var("var_4188", dtype = "float64", shape = (1280,))#candidate|4188|(1280,)|var|float64
var_4189 = relay.var("var_4189", dtype = "uint16", shape = (336,))#candidate|4189|(336,)|var|uint16
call_4186 = func_4185_call(var_4187,var_4188,var_4189,)
output = call_4186
func_4190 = relay.Function([var_4187,var_4188,var_4189,], output)
mutated_mod['func_4190'] = func_4190
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4415 = relay.var("var_4415", dtype = "float64", shape = (7, 7, 9))#candidate|4415|(7, 7, 9)|var|float64
uop_4416 = relay.sqrt(var_4415.astype('float64')) # shape=(7, 7, 9)
uop_4424 = relay.cos(uop_4416.astype('float64')) # shape=(7, 7, 9)
func_1747_call = mod.get_global_var('func_1747')
func_1749_call = mutated_mod.get_global_var('func_1749')
var_4435 = relay.var("var_4435", dtype = "int8", shape = (729,))#candidate|4435|(729,)|var|int8
call_4434 = relay.TupleGetItem(func_1747_call(relay.reshape(var_4435.astype('int8'), [9, 9, 9])), 0)
call_4436 = relay.TupleGetItem(func_1749_call(relay.reshape(var_4435.astype('int8'), [9, 9, 9])), 0)
output = relay.Tuple([uop_4424,call_4434,var_4435,])
output2 = relay.Tuple([uop_4424,call_4436,var_4435,])
func_4455 = relay.Function([var_4415,var_4435,], output)
mod['func_4455'] = func_4455
mod = relay.transform.InferType()(mod)
var_4456 = relay.var("var_4456", dtype = "float64", shape = (7, 7, 9))#candidate|4456|(7, 7, 9)|var|float64
var_4457 = relay.var("var_4457", dtype = "int8", shape = (729,))#candidate|4457|(729,)|var|int8
output = func_4455(var_4456,var_4457,)
func_4458 = relay.Function([var_4456,var_4457,], output)
mutated_mod['func_4458'] = func_4458
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4545 = relay.var("var_4545", dtype = "float64", shape = (3, 13, 14))#candidate|4545|(3, 13, 14)|var|float64
uop_4546 = relay.sqrt(var_4545.astype('float64')) # shape=(3, 13, 14)
var_4559 = relay.var("var_4559", dtype = "float64", shape = (3, 13, 14))#candidate|4559|(3, 13, 14)|var|float64
bop_4560 = relay.right_shift(var_4545.astype('uint16'), relay.reshape(var_4559.astype('uint16'), relay.shape_of(var_4545))) # shape=(3, 13, 14)
output = relay.Tuple([uop_4546,bop_4560,])
output2 = relay.Tuple([uop_4546,bop_4560,])
func_4590 = relay.Function([var_4545,var_4559,], output)
mod['func_4590'] = func_4590
mod = relay.transform.InferType()(mod)
var_4591 = relay.var("var_4591", dtype = "float64", shape = (3, 13, 14))#candidate|4591|(3, 13, 14)|var|float64
var_4592 = relay.var("var_4592", dtype = "float64", shape = (3, 13, 14))#candidate|4592|(3, 13, 14)|var|float64
output = func_4590(var_4591,var_4592,)
func_4593 = relay.Function([var_4591,var_4592,], output)
mutated_mod['func_4593'] = func_4593
mutated_mod = relay.transform.InferType()(mutated_mod)
const_4749 = relay.const([[[8.297564,6.256575,-9.091508,7.744713,2.508269,5.239087,-0.185273,0.326047,8.545620,-0.275029]],[[-1.615208,2.800493,-5.950146,9.754130,9.312699,4.385004,5.084068,-0.703568,-8.133540,2.684144]],[[0.806804,-1.123654,4.641884,-4.859418,7.189492,-2.182708,9.263260,0.754791,-5.114970,-3.794037]],[[1.236425,7.863170,-8.940461,-5.909625,8.767583,6.037754,6.496567,-6.669673,5.057374,4.909757]],[[4.136188,-0.372236,1.005302,-8.658072,-4.724698,2.125152,-4.910313,2.118988,9.621345,8.705410]],[[3.344739,6.444289,-8.992634,0.775415,-7.770224,2.998077,8.074880,9.374610,8.264357,-0.672031]],[[-7.497327,5.434209,-2.326643,-7.277837,-8.936282,-9.512998,-7.200286,-3.146617,3.985443,-6.373790]],[[-5.913788,8.982445,8.708646,4.756904,-9.530218,-3.214226,0.793491,-0.755970,-5.577964,8.745012]],[[-3.253130,2.923107,7.605901,-7.074039,7.587093,4.117800,-7.452706,8.415072,0.488939,-6.023878]],[[8.955901,-2.973071,-4.089047,-4.984805,7.105718,4.071740,-4.148550,-7.184928,5.155602,-9.459078]],[[5.174797,-2.951206,5.671904,9.865787,-1.644815,-0.086367,6.882172,8.092513,8.779675,-9.826723]],[[3.582020,3.907982,-0.181652,-4.123219,-5.759564,5.138537,-6.784468,2.394877,1.454101,-2.116283]],[[-4.313406,3.741634,2.735315,2.499440,6.488172,-0.747852,6.599107,-9.039532,3.775926,5.514466]],[[3.648137,-3.564463,8.240770,0.423174,-5.148639,-1.736432,8.334390,-9.596985,-8.092245,4.566529]]], dtype = "float32")#candidate|4749|(14, 1, 10)|const|float32
const_4750 = relay.const([[[2.099914,0.430746,-6.741536,-9.287980,7.849243,2.156140,-5.874823,-0.100483,-9.851115,6.584419],[-3.994788,-0.857074,7.755123,-1.448018,2.192702,-8.381725,1.416164,-8.108072,-8.355027,-3.330412]],[[4.716588,-0.454809,-8.757094,-5.076571,-4.572047,-3.335101,-6.760141,0.273495,8.789190,8.150417],[8.935503,3.189674,0.806290,-6.525016,4.092530,-8.366226,-4.619495,-8.681172,6.951867,1.392300]],[[2.804423,-6.126340,1.941673,6.239916,-5.335357,7.981827,-6.610807,-6.196561,-9.354637,-2.926857],[1.717320,-3.288203,8.598729,3.673902,-9.662389,-4.012034,-3.828071,4.163859,9.327746,-6.960426]],[[4.571249,2.377076,3.609501,6.250338,3.858208,7.672109,-4.494475,-4.152330,-9.597460,-9.400008],[0.349811,-7.549246,-0.473280,8.115513,1.051708,4.594961,0.688213,-6.906571,-4.692188,0.070965]],[[2.060227,1.683562,-6.922366,-1.374871,9.892254,9.028712,9.755803,-7.530086,2.635816,5.449268],[-5.328799,8.182038,-7.572744,-4.760253,-7.864890,-6.115578,-0.995030,1.587879,9.638742,6.513881]],[[6.104378,2.617393,5.362507,-2.437719,0.732736,6.015428,-4.320000,-8.797652,-7.881159,7.578841],[-7.501291,3.124943,9.430752,7.766252,-8.563385,4.548189,-6.761321,-9.286220,4.091084,-8.447112]],[[1.421671,-1.876989,2.365275,-0.242242,1.498286,-8.693469,1.431633,-3.408768,-4.632990,2.561048],[-5.897852,-6.139779,-9.338596,-3.493324,0.317629,-6.534552,-1.617044,-8.259036,7.347408,-7.250509]],[[2.043324,-7.484636,-7.451041,9.966288,5.091197,8.302908,-4.878938,-7.458178,-8.295440,-7.125967],[4.440608,-9.818986,0.920196,8.003437,6.692724,6.632098,-4.529506,3.629405,-0.549779,-1.101943]],[[-5.996008,3.581944,3.224522,4.665457,3.479618,-3.370482,-7.288154,-7.875462,7.932636,5.298878],[2.420802,6.537956,-2.393860,-3.630213,-4.688011,-9.336762,-7.298299,6.096518,0.525551,-5.371245]],[[-7.475621,-5.841172,8.016436,-3.284403,-4.779474,-5.649827,-9.245732,-3.036969,-7.215741,9.285932],[5.197226,2.405939,1.948615,6.772341,3.340107,4.815830,-0.918192,5.612692,8.883599,3.938525]],[[6.282351,6.624082,6.995543,-0.427663,0.963324,-5.539002,-4.186675,-6.411727,-4.654819,-3.800259],[7.043373,2.614417,9.784658,-6.737028,-2.750986,1.868958,2.487492,2.885120,-2.217860,-7.046684]],[[6.231720,-2.448036,7.080085,-9.018410,-2.781066,1.084802,-6.468798,-0.928550,9.577887,-6.485833],[-8.060940,2.685145,7.796585,-1.965577,9.785839,1.649575,-2.997721,-6.931684,-9.301372,7.534799]],[[1.132456,5.177845,-5.418378,-6.651550,-3.450845,6.340069,-2.743037,-8.546520,-7.292285,7.193669],[7.083004,-2.924908,-4.321056,5.042197,9.769025,5.582340,-3.920682,7.137254,-3.841324,-4.781780]],[[-4.086404,9.397056,5.722467,-8.462658,9.990856,0.506013,-5.230292,-0.702629,-5.810298,-4.196339],[-0.773643,-0.740477,-0.604525,0.024436,2.833641,2.315230,-5.515716,-8.320707,-0.699688,6.459263]]], dtype = "float32")#candidate|4750|(14, 2, 10)|const|float32
bop_4751 = relay.floor_mod(const_4749.astype('float32'), const_4750.astype('float32')) # shape=(14, 2, 10)
func_2996_call = mod.get_global_var('func_2996')
func_3000_call = mutated_mod.get_global_var('func_3000')
const_4760 = relay.const([0.516482,4.503983,4.377937,6.654346,4.362796,-3.501732,-2.338172,-6.785860,3.142822,-5.875358,-4.549213,5.095046,-0.685172,-8.505727,2.785850,5.406962,-0.587862,9.052168,-7.915521,2.638081,-9.808752,5.529856,-1.193110,8.798745,-0.428193,6.083015,1.091793,5.856247,-0.194583,-5.412163,7.462395,-6.385492,-9.769241,6.758445,6.319692,-8.229659,9.894282,-7.484652,4.422693,7.265396,6.005165,3.261105,-1.468713,-5.608407,-8.904836,-2.715189,-1.698186,4.798788,-4.519019,-0.014247,7.248054,0.797860,-1.872871,3.997450,8.616020,5.829532,2.963026,3.683631,9.744002,5.281396,-8.986068,0.047529,-7.602679,-1.476074,-0.665043,-7.182631,8.866312,-2.400341,9.516760,-3.359588,-1.213762,-6.305388,-0.508141,4.155138,3.292566,1.258034,-6.542662,1.130569,5.412150,4.768206,-7.554682,-8.623478,4.130814,1.843354,-8.262791,3.335489,-1.585311,-0.945524,-5.749140,-3.433833,-3.086105,5.220489,4.136595,-6.151358,-7.913257,-3.786898,6.805512,6.856030,8.775919,-4.254081,-2.288719,-6.622074,2.064094,9.721268,7.669949,4.236780,8.026625,-3.342159,8.827446,2.734590,5.979509,-9.465435,-6.389981,6.307965,6.619777,7.484431,-5.758148,-6.943126,2.430692,4.532678,8.570071,-7.968457,5.044010,0.654473,7.262998,4.373576,-8.384546,-8.763339,7.208010,-5.223453,3.761691,-7.848750,-2.152909,9.090903,-0.339900,-9.931135,4.587001,-2.051805,7.976611,-1.825104,-2.748753,6.573262,2.020172,-7.653483,-3.060007,0.888084,-8.897669,4.925092,-8.877670,-5.228153,4.896791,0.540435,-2.731257,7.669018,2.410941,-1.853916,5.506099,4.946454,4.931367,6.463228,-1.758288,1.756057,-8.653938,0.170911,6.816881,2.465511,-8.366737,-3.202333,-6.942616,8.336821,8.265916,-5.376007,2.058558,8.136346,1.434401,-9.887480,-2.815301,-2.363889,-7.365632,7.865061,-2.770588,-3.076187,6.887116,8.485172,7.821117,-9.441345,-5.706189,7.983371,-3.468054,6.847753,-5.618431,2.394597,2.177147,7.777705,-8.895320,-6.679773,-2.378134,7.383982,2.101982,3.221351,1.597506,9.790035,-7.183624,-7.555576,6.175305,-8.063258,3.504536,-9.163762,-0.077206,8.536410,1.359794,7.963289,-7.073934,6.713951,4.450899,-6.216575,-0.405293,-1.077138,6.576247,2.983805,-2.395301,-5.827512,-6.507700,9.081436,-5.608415,5.238873,-5.622437,5.029817,5.456446,-8.909385,-3.611379,-3.628935,-1.186837,2.759762,6.144857,-2.220364,4.645625,5.928291,-8.568500,-9.541576,2.687505,-4.866290,-4.874466,9.271300,-3.368081,-1.800769,-6.462991,-5.948074,1.943362,-5.123166,-0.080692,4.716556,6.749259,5.125735,5.437119,5.675869,8.695848,-5.088208,7.005900,-8.389548,8.859899,-3.018378,9.569230,-5.047251,6.006880,9.502548,8.960165,1.310252,6.176320,-9.426072,-8.516153,-4.971696,4.977192,0.921296,-9.972170,1.666633,-8.138693,2.848876,-8.889326,-1.148666,7.753089,0.115804,-1.629643,1.493260,-0.127751,-4.984810,6.036939,-6.543116,6.164220,-6.536433,9.454938,0.753329,3.777347,5.177448,-4.827389,2.306746,6.990595,3.167263,5.410546,-6.085183,-2.613632,-4.637214,8.534169,1.742356,-0.123846,-6.622331,7.015276,5.233658,-0.257467,-6.006497,8.980086,-0.073018,-7.838344,-3.644115,9.547302,1.709895,-5.907455,1.617107,3.262701,-4.142491,-5.893361,3.806963,6.255705,5.644785,-4.308603,-4.970576,0.684310,-0.579010,-8.820924,-6.343823,7.012966,-6.993081,0.834280,1.577841,-0.527552,7.307563,-7.250532,-9.408751,3.820775,-7.553635,7.915805,3.393022,3.073144,-0.130029,1.563960,5.073191,5.219451,-3.241913,-1.668229,7.357547,4.827903,7.161833,9.165420,5.797397,1.728011,-5.956375,8.896965,-1.466042,-1.607276,-9.805391,8.678308,-6.679718,8.501115,4.628461,3.156200,-3.580076,8.955463,-5.905818,1.110175,-2.478619,4.837432,8.623939,-8.123085,-9.512922,1.641637,-4.363284,-2.018282,5.881921,5.579612,1.769275,6.824628,-7.207159,-2.675213,4.557394,-4.318119,-5.393034,6.705895,-9.039671,8.331871,-6.848168,-2.023204,0.592199,0.478475,-8.819987,-5.029517,8.204932,-1.040485,-7.995383,8.096697,-6.736000,5.507058,4.314166,6.847442,3.878538,-5.252420,7.823174,-9.157117,-6.351557,9.601203,0.707709,-8.115171,5.553197,-9.188357,6.201915,8.728621,3.207121,9.140574,2.455206,1.262861,2.012662,-5.102932,-4.023404,-8.440054,7.976491,2.026331,4.295528,9.724774,-1.228127,-7.212718,9.059624,-9.197012,3.503589,-3.472484,-1.147139,4.508187,9.532657,-3.004453,-6.330965,-2.126658,-1.191315,-2.734445,-6.821728,-6.580445,3.823254,-8.935689,-5.777886,-4.532645,-7.438200,8.966842,-0.438351,1.379221,-5.694844,4.730235,-0.413686,-5.968292,1.883136,-2.627126,4.674455,4.568258,5.380329,2.640694,-4.496170,5.553909,-9.150536,5.308857,5.865301,6.313688,-6.439922,2.589383,-7.931373,8.603524,-4.619798,3.509589,-3.538418,2.992308,2.064108,1.205208,-6.191321,-9.499224,1.011435,2.977765,-4.491304,0.383660,9.356058,2.173737,-9.941669,1.765707,-3.317725,-1.806531,8.245949,5.983861,9.141806,-6.055962,1.362799,-1.668171,-6.163299,-2.988703,-5.131473,3.728374,-8.040969,-8.662867,3.424492,8.726836,-9.498231,-3.430364,6.107344,-0.469300,6.836816,-4.849174,-7.184507,7.429215,-6.801452,9.891693,7.203923,-0.727366,-7.602745,2.035335,5.724541,1.840723,-0.467172,1.787310,1.806088,-3.343905,5.498755,-8.157904,-0.291494,-2.510666,1.839361,0.353777,8.031446,-2.848894,-2.382194,7.489809,-7.839589,-3.249978,4.712859,9.933901,7.447336,2.578249,-1.078095,-5.360314,2.257251,-0.805139,9.522185,-5.222785,2.696357,-3.832467,-3.121216,0.483183,-2.452253,7.071313,-0.294835,2.646797,2.166431,-7.355281,-1.838271,-2.494871,8.237333,-1.158945,-7.535823,1.784999,7.685999,-7.753256,-7.070087,-5.461196,5.929654,-7.380724,-8.120214,9.183052,-1.481910,-5.705437,5.424971,-6.639896,1.500408,0.557044,2.745369,4.693455,-1.418206,-0.254784,-9.938729,2.484228,1.067245,4.836953,8.042727,5.585881,-7.056776,-6.949109,-0.073287,-0.315249,3.134050,8.621840,7.290732,7.220549,-5.160666,-7.222946,4.439709,-9.965452,3.714453,-0.866173,0.236151,-3.253699,1.342614,9.175845,4.852142,2.510677,4.069511,-9.324138,-4.070401,-9.213178,5.450175,2.220300,-1.096569,9.986646,5.157162,6.586381,9.797196,-4.734687,-2.293914,-7.368167,8.867346,0.215173,5.521901,6.832684,9.325313,-8.910968,-8.681543,6.455291,-9.479335,-1.705574,6.730888,-6.463374,9.874071,7.513698,-3.822698,-4.717167,-4.648103,-2.988571,7.402397,6.660835,9.545862,7.129366,5.538839,-4.209120,-2.523442,5.792238,-3.592800,-6.646388,-0.615202,-4.590254,-7.557693,-8.575932,-4.945429,-0.595710,0.960550,-1.257795,9.039648,-9.033452,0.159214,-3.048531,-5.747071], dtype = "float64")#candidate|4760|(660,)|const|float64
const_4761 = relay.const(2, dtype = "uint8")#candidate|4761|()|const|uint8
call_4759 = relay.TupleGetItem(func_2996_call(relay.reshape(const_4760.astype('float64'), [11, 6, 10]), relay.reshape(const_4760.astype('float64'), [11, 6, 10]), relay.reshape(const_4761.astype('uint8'), []), ), 3)
call_4762 = relay.TupleGetItem(func_3000_call(relay.reshape(const_4760.astype('float64'), [11, 6, 10]), relay.reshape(const_4760.astype('float64'), [11, 6, 10]), relay.reshape(const_4761.astype('uint8'), []), ), 3)
output = relay.Tuple([bop_4751,call_4759,const_4760,const_4761,])
output2 = relay.Tuple([bop_4751,call_4762,const_4760,const_4761,])
func_4770 = relay.Function([], output)
mod['func_4770'] = func_4770
mod = relay.transform.InferType()(mod)
mutated_mod['func_4770'] = func_4770
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4770_call = mutated_mod.get_global_var('func_4770')
call_4771 = func_4770_call()
output = call_4771
func_4772 = relay.Function([], output)
mutated_mod['func_4772'] = func_4772
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4770_call = mod.get_global_var('func_4770')
func_4772_call = mutated_mod.get_global_var('func_4772')
call_4805 = relay.TupleGetItem(func_4770_call(), 0)
call_4806 = relay.TupleGetItem(func_4772_call(), 0)
func_2996_call = mod.get_global_var('func_2996')
func_3000_call = mutated_mod.get_global_var('func_3000')
const_4812 = relay.const([0.160636,-8.399228,-6.880547,-7.600856,2.136382,-0.745218,-0.080293,8.944891,0.480972,2.966657,5.348058,3.806903,-4.910892,3.943544,4.606746,0.779281,3.288288,-3.904441,8.658911,-6.387666,-1.750878,6.975527,-1.124591,1.911917,6.627680,1.111478,5.874316,-2.908751,0.013403,5.380378,-8.387887,2.581435,-4.766847,-4.896537,-1.193794,-3.630945,7.191687,-1.266234,1.573067,2.180022,-6.740227,6.002326,2.005422,-3.261789,6.235975,-3.260959,-2.511359,8.067346,3.180739,3.736451,-1.284330,-9.481157,8.508764,1.381255,-4.220117,7.770270,5.262573,-7.407735,0.280899,-7.653837,-7.251897,-7.089058,-3.622943,2.639365,2.004336,-3.011927,-9.352564,7.630313,9.633020,-4.173447,1.121373,-2.322779,-9.087651,-3.937723,5.617949,-5.667922,6.973694,9.248443,5.219725,4.880136,9.604387,-9.944036,-3.850675,4.823293,1.744121,1.614495,3.814617,6.800121,-6.333460,2.971178,-0.415033,5.955991,5.499937,-0.337325,0.905524,2.781308,9.270742,-6.770518,-2.833696,4.716798,-2.395704,-9.882753,-6.690685,6.883877,3.364970,3.095175,3.878650,4.581454,6.622126,0.999947,9.050227,0.818211,-6.427889,-8.885670,5.422465,-6.402590,-9.939976,9.178369,9.201095,1.768473,-9.440549,-9.282811,0.831281,-0.245379,9.966737,6.578898,-7.849564,0.038859,5.399261,3.674981,9.554737,-2.080548,9.523877,7.487249,-3.360186,2.723429,5.857270,-7.770450,-0.888979,4.077031,-2.283409,-2.704980,4.058743,-9.317097,-1.917056,8.076395,8.166278,6.805413,9.893054,8.098138,-3.287754,-0.617265,9.160056,-1.212640,6.092732,1.633631,2.322189,0.352078,-2.106887,7.386124,8.303422,-3.183978,-2.990689,3.727650,-5.465812,8.227075,-0.594197,-4.050982,9.592738,5.974178,-3.380014,-5.379828,-1.702453,-4.710917,6.477783,4.001897,-4.978871,-5.258567,-3.817796,1.085743,-2.392530,0.323870,-1.593047,-9.190138,-4.450485,8.019518,5.869340,-5.015659,7.314501,-6.963981,-7.470766,4.591965,2.687752,6.046942,-7.813422,1.599542,-5.547467,4.551905,-8.440577,-5.098068,8.078413,-9.418493,5.474342,-0.841647,9.547831,-4.889551,3.061306,2.099129,3.160034,-6.145320,1.907464,-3.516049,-6.020894,-5.484739,-5.601378,1.450935,-7.800227,-3.159590,0.436440,-3.834663,0.864546,-6.429674,1.485295,2.731003,8.963872,-4.165199,8.810593,-5.951942,-7.507825,8.702715,7.012734,-3.572965,-8.887014,1.792100,6.878769,-9.035934,-3.128056,8.788842,-8.541645,-8.235022,-0.893943,-4.938805,-0.834648,8.740890,-2.229302,3.488445,-4.290152,5.747920,-6.192905,-2.355896,4.668240,-1.273258,9.565021,5.379472,-2.562834,2.408596,-5.718530,5.986974,-9.222541,2.080271,4.173308,-6.221139,-0.882703,-7.689358,6.309121,-5.494765,9.983170,-5.427900,8.561404,4.542439,-6.191610,-3.387464,-7.322964,6.872287,6.549985,4.292163,0.476036,-9.697145,-7.020507,-8.797717,3.096131,-9.257476,6.459017,0.991691,-3.945353,-7.529324,-4.577382,9.887217,-4.592008,4.072333,6.696225,-8.483435,2.539652,-7.700787,4.123972,0.520535,7.988621,-4.849262,-6.987340,4.309594,-3.863241,2.935807,6.375518,4.291644,6.786988,4.534290,-0.403550,9.928781,-2.471496,2.568200,-7.850333,-6.323400,-7.860968,-7.132458,-3.402379,6.346300,-0.402731,4.096566,0.840286,8.813794,-2.822889,8.147721,5.652569,-2.620536,-0.923231,9.258401,-0.219240,-6.913838,-9.286483,1.907389,5.468748,5.881288,-6.221470,8.241137,-2.323071,-5.359211,0.798615,2.808932,9.352165,-0.185520,5.195982,-2.621155,1.819736,-8.639437,-2.762303,-0.191036,-7.368768,0.781942,2.425598,-4.146521,0.312984,-2.206906,-7.124501,5.043859,4.433546,6.244293,-4.605074,-1.642209,-6.872654,2.769805,-7.908131,-0.046544,-8.214457,2.745779,-1.798598,4.854066,8.686645,-6.562091,6.894688,9.834094,8.704291,7.643691,-7.136683,3.811099,-1.001569,1.130189,8.590428,5.887176,0.886832,0.624364,-1.137106,-4.708882,-4.550263,8.592701,-5.963372,-2.796621,5.298016,5.016530,5.440800,-2.263476,-9.529199,-9.737706,-8.709580,2.480013,4.519654,5.377885,1.909081,2.713022,0.513974,0.936474,3.891683,5.368563,7.275853,-5.538755,-0.316331,-8.126768,7.945831,-3.995532,-5.471133,7.556757,-8.770234,4.046669,-5.281172,4.595364,7.322942,-0.257521,-7.556199,8.391335,-9.481379,-2.099245,-8.815982,2.807427,9.707452,8.836004,9.286871,6.526776,-0.589995,1.124494,9.326238,-6.384335,1.510160,1.427026,-3.589319,2.184634,6.869101,4.849867,-5.701840,2.573863,-7.529475,4.261131,0.257918,8.704885,6.973314,0.657892,4.125438,7.099599,-6.085989,-0.538044,1.633959,6.217416,2.047934,9.065077,-4.621524,-4.881960,6.185990,-6.492952,-5.233040,-5.680701,6.509748,4.579131,0.646905,8.986841,0.951338,2.219244,-8.977172,6.965615,3.676594,3.469387,1.843814,8.992901,-8.021955,-1.427360,4.207929,-6.269986,-2.369465,-8.700779,4.510461,7.256098,1.370114,6.123379,4.915678,-7.604743,3.441991,-1.779322,2.568693,-5.939068,-5.950668,-3.430576,1.400016,-8.514806,-4.509062,-7.713214,-8.896201,9.842813,-1.367863,1.616749,3.952493,-8.118898,-7.287995,-3.064316,-3.644309,1.517451,2.235410,2.614833,-3.125675,-8.514018,-9.244255,-8.806728,-6.212881,-9.415629,1.784950,-5.043899,5.690174,4.019226,-5.546753,7.243606,-6.862959,0.170456,8.191098,1.258847,-5.055773,-5.746475,-6.373757,5.361004,-9.418046,7.829027,-0.602106,6.134684,-5.058045,6.019775,-9.590749,-8.238932,3.171133,-5.819896,-2.373434,-1.981431,0.555125,2.683647,-2.042919,-6.320102,-2.652461,-9.582647,2.647504,0.193326,-9.238498,-1.722062,4.466541,-8.047160,-0.209865,7.147742,-8.633912,6.575862,3.109396,3.197164,-6.509382,-9.574627,7.969987,6.132122,-8.189447,-2.263113,4.007839,0.634084,7.565961,8.405603,-4.904129,-7.422545,-6.607948,-5.422758,-3.010008,5.943857,7.659206,-6.874537,-6.139224,-1.508343,-0.062923,-9.467509,4.854239,-3.981112,6.093799,4.560826,2.373659,-8.075598,2.310389,3.526455,-5.749384,-6.904985,-2.188822,-2.239142,-1.792583,-2.249824,4.496825,-3.470504,-2.556219,-8.432167,-6.082485,-4.503962,3.652343,-1.575932,0.994645,2.303712,-1.842348,-3.958538,-8.488286,-2.743073,0.442845,5.438037,6.833454,8.768077,-9.940879,8.292323,8.468396,-2.392235,-8.187163,-9.211916,-4.730725,9.019222,9.618768,-9.096216,-9.089420,4.590909,2.412201,-1.702639,-8.332193,-9.481969,7.861415,-8.861524,7.347186,-0.356187,5.030342,-8.388811,-1.424688,-0.940829,5.856948,3.399774,7.498817,1.169972,4.756133,1.097492,3.803626,-0.241686,9.677628,-0.268490,-9.433238,-1.531852,9.019377,0.884659,-0.775192,6.146747,-6.696252,5.080835,5.977826,-3.450897,0.076423,2.242359,-0.187109,1.046751,-3.923464,-1.665121,-2.907050,-1.057630], dtype = "float64")#candidate|4812|(660,)|const|float64
const_4813 = relay.const(1, dtype = "uint8")#candidate|4813|()|const|uint8
call_4811 = relay.TupleGetItem(func_2996_call(relay.reshape(const_4812.astype('float64'), [11, 6, 10]), relay.reshape(const_4812.astype('float64'), [11, 6, 10]), relay.reshape(const_4813.astype('uint8'), []), ), 0)
call_4814 = relay.TupleGetItem(func_3000_call(relay.reshape(const_4812.astype('float64'), [11, 6, 10]), relay.reshape(const_4812.astype('float64'), [11, 6, 10]), relay.reshape(const_4813.astype('uint8'), []), ), 0)
func_2996_call = mod.get_global_var('func_2996')
func_3000_call = mutated_mod.get_global_var('func_3000')
call_4831 = relay.TupleGetItem(func_2996_call(relay.reshape(call_4811.astype('float64'), [11, 6, 10]), relay.reshape(const_4812.astype('float64'), [11, 6, 10]), relay.reshape(const_4813.astype('uint8'), []), ), 1)
call_4832 = relay.TupleGetItem(func_3000_call(relay.reshape(call_4811.astype('float64'), [11, 6, 10]), relay.reshape(const_4812.astype('float64'), [11, 6, 10]), relay.reshape(const_4813.astype('uint8'), []), ), 1)
output = relay.Tuple([call_4805,call_4811,const_4812,const_4813,call_4831,])
output2 = relay.Tuple([call_4806,call_4814,const_4812,const_4813,call_4832,])
func_4857 = relay.Function([], output)
mod['func_4857'] = func_4857
mod = relay.transform.InferType()(mod)
output = func_4857()
func_4858 = relay.Function([], output)
mutated_mod['func_4858'] = func_4858
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4857_call = mod.get_global_var('func_4857')
func_4858_call = mutated_mod.get_global_var('func_4858')
call_4865 = relay.TupleGetItem(func_4857_call(), 0)
call_4866 = relay.TupleGetItem(func_4858_call(), 0)
output = call_4865
output2 = call_4866
func_4889 = relay.Function([], output)
mod['func_4889'] = func_4889
mod = relay.transform.InferType()(mod)
mutated_mod['func_4889'] = func_4889
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4889_call = mutated_mod.get_global_var('func_4889')
call_4890 = func_4889_call()
output = call_4890
func_4891 = relay.Function([], output)
mutated_mod['func_4891'] = func_4891
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4889_call = mod.get_global_var('func_4889')
func_4891_call = mutated_mod.get_global_var('func_4891')
call_4923 = func_4889_call()
call_4924 = func_4889_call()
func_4185_call = mod.get_global_var('func_4185')
func_4190_call = mutated_mod.get_global_var('func_4190')
var_4928 = relay.var("var_4928", dtype = "float32", shape = (792,))#candidate|4928|(792,)|var|float32
var_4929 = relay.var("var_4929", dtype = "float64", shape = (2, 640))#candidate|4929|(2, 640)|var|float64
var_4930 = relay.var("var_4930", dtype = "uint16", shape = (336,))#candidate|4930|(336,)|var|uint16
call_4927 = relay.TupleGetItem(func_4185_call(relay.reshape(var_4928.astype('float32'), [8, 11, 9]), relay.reshape(var_4929.astype('float64'), [1280,]), relay.reshape(var_4930.astype('uint16'), [336,]), ), 3)
call_4931 = relay.TupleGetItem(func_4190_call(relay.reshape(var_4928.astype('float32'), [8, 11, 9]), relay.reshape(var_4929.astype('float64'), [1280,]), relay.reshape(var_4930.astype('uint16'), [336,]), ), 3)
func_4105_call = mod.get_global_var('func_4105')
func_4109_call = mutated_mod.get_global_var('func_4109')
var_4933 = relay.var("var_4933", dtype = "float64", shape = (36,))#candidate|4933|(36,)|var|float64
call_4932 = func_4105_call(relay.reshape(var_4933.astype('float64'), [12, 3, 1]), relay.reshape(var_4933.astype('float64'), [12, 3, 1]), )
call_4934 = func_4105_call(relay.reshape(var_4933.astype('float64'), [12, 3, 1]), relay.reshape(var_4933.astype('float64'), [12, 3, 1]), )
func_1497_call = mod.get_global_var('func_1497')
func_1499_call = mutated_mod.get_global_var('func_1499')
const_4946 = relay.const([-5.959530,9.410111,4.043727,-3.491756,-4.123064,9.245405,-2.600692,-1.671399,7.327131,6.049444,-0.406046,-8.175529,4.109548,1.039814,-6.540216,6.848327,7.019844,9.556094,-0.643884,-4.806477,1.652299,1.397858,-5.902582,-9.101253,-9.932031,-7.969307,-8.109687,6.461335,-5.605952,-2.765012,2.523255,2.687632,-0.994960,9.953450,-4.586845,6.134524,-9.867997,8.944509,8.318229,-2.638093,-6.922103,-7.953031,4.896408,-6.417235,7.044160,-8.533466,-7.811017,3.623975,-0.145279,-3.823700,-3.716370,0.171994,-0.535900,6.404054,-1.206268,0.457460,-7.008277,-5.706846,-3.682866,0.309686,7.099258,-8.215897,-5.204118,1.856476,-8.646765,6.370054,7.819695,8.521493,-8.055123,2.321863,-7.629814,1.090416,4.170180,-8.913190,-7.504564,0.332330,3.600057,-9.271027,9.171053,0.777310,6.561971,2.342794,7.272457,-5.674860,2.422958,-7.712614,-8.387552,6.295770,3.657884,1.404461,8.688422,-5.548064,-4.578615,-0.565926,-8.919504,-9.984171,8.314915,-7.148745,-0.314939,-9.492625,0.556890,-7.293507,-8.818000,9.004907,6.893656,5.382256,8.785066,-4.462936,-4.396200,1.977810,-9.307399,-2.510378,6.880024,-6.106401,2.421041,-9.358131,6.677641,3.155616,-5.882813,-8.290651,6.918115,-9.602554,4.207963,1.403638,-9.411859,-8.782225,-7.147283,-3.140977,7.661683,0.643727,1.872057,7.187002,-1.238993,-4.881754,5.760710,3.978925,-4.669767,6.984109,-5.266288,-7.648826,-7.961673,6.863648,3.347559,-1.332094,-6.162341,-0.615952,9.404881,-2.636934,-1.037455,-6.073290,5.114082,-5.185977,-9.826572,8.785210,-9.829895,5.400027,-2.698031,7.720239,-3.828220,5.866081,-0.976009,-6.880475,3.574966,4.415323,8.527060,-3.154604,2.860638,-2.940098,-7.738279,-5.845901,3.473283,8.935239,-6.402399,-2.005171,7.325381,3.943430,2.778220,5.367377,4.627141,-3.708905,2.768908,2.023526,0.190831,-3.997711,-5.329771,6.020601,-2.360840,6.772978,0.365546,3.905950,4.876443,-8.447764,-4.413480,-5.801063,7.996956,-9.199455,0.859278,5.019093,7.213304,-1.626522,2.251093,-4.008701,8.993164,-5.956991,6.898715,7.504945,4.515978,1.009702,1.027712,7.330820,-2.210308,8.801553,6.506576,-1.469790,-3.621701,-4.760803,-5.223673,9.973333,1.861498,4.459931,1.632167,4.784540,-2.180065,-8.868610,-3.299773,-9.605872,3.720077,-6.908409,2.409363,-2.537682,-0.028642,3.350937,1.868464,-8.323241,-1.692996,7.219933,-4.223625,-7.345072,2.582171,-0.794549,-9.758410,-2.534533,1.923299,3.205826,7.723005,-5.730265,-9.177085,1.786325,5.842671,-0.549505,7.583050,9.428819,9.580197,-9.322736,5.791735,-9.752169,-7.526943,5.247119,-8.330632,-9.315030,-4.650556,5.423385,-5.056081,-3.595597,-1.862791,8.250781,-4.170873,8.483821,1.355566,-3.656556,-8.152731,-6.660832,1.160849,-8.387512,-5.413461,-4.934186,-4.409844,-2.655526,-9.940716,-7.195784,1.625016,-6.424829,-9.470585,5.176163,9.413591,-9.626623,7.581836,-5.593511,5.005189,-7.958503,3.574609,-4.105090,-6.189488,-4.885011,-4.308891,1.668653,5.087922,2.377660,7.060035,0.834502,4.783040,-3.214431,-6.205019,7.961611,-5.274401,-9.170608,-7.942243,7.254626,3.295769,5.864818,-5.137442,5.523026,-7.192187,5.558673,-8.404355,5.042546,5.090103,-4.444838,-1.633815,4.777495,6.420572,8.322617,-8.240173,-5.494041,-1.839207,1.631962,-7.597302,-4.585724,-3.929522,-4.890657,-0.631609,-3.022738,9.165185,-9.070591,-9.977885,-7.023859,4.085472,-3.313006,-3.263954,8.269753,-4.817745,1.338730,5.561516,-5.724453,-0.349377,-2.696704,-3.229753,-0.791317,9.927860,7.492962,-9.232271,9.791286,2.307290,1.688387,-5.491573,7.439658,-4.802060,3.300619,-7.180053,7.041332,-1.357050,-5.347253,-7.226175,7.695596,5.065829,9.364384,-8.169270,8.531811,-5.968913,-6.068997,0.494163,-8.038291,-8.688211,5.961351,5.141498,9.490898,9.302583,-9.486491,-2.527848,7.012031,-3.343996,4.809285,-6.929334,5.601383,-8.007820,2.966637,7.346486,5.612277,-0.502571,-2.087908,5.901572,6.877914,7.508879,-9.788308,3.430757,0.788766,-2.436961,5.589290,9.461563,-6.710628,-4.987318,5.465522,8.887897,2.665620,-4.680916,5.698304,-6.899267,-4.783597,-4.671238,7.559688,-1.039102,0.671526,-1.051586,7.574169,0.692575,4.009305,-3.427705,3.684870,3.470747,8.094877,1.071724,7.835594,9.171642,9.766129,-8.984075,-9.652939,-2.692380,8.653868,-5.111407,-2.706320,-7.529988,4.493160,3.958856,-1.921581,4.478773,-3.340734,-6.831355,-1.586361,6.266170,8.192382,3.047471,-0.998676,-3.120942,-5.488791,9.242681,-5.494353,-5.924286,-4.812805,8.168079,-6.597993,-9.015037,3.432916,-9.180166,-5.603480,3.838814,4.396784,3.098105,-1.036798,2.335473,-9.486288,-4.985335,2.277487,-8.292212,-0.764992,-0.652124,1.144839,0.607202,-7.970967,-8.542131,6.578877,-5.372880,-0.187853,-8.402463,7.886430,0.874580,-2.260224,4.983694,-6.085637,4.515797,9.567663,8.466469,-4.644373,3.875040,-3.673506,-4.935271,5.411518,-5.138386,-6.962597,2.057498,5.631083,-0.404951,-2.157874,3.102209,0.919338,5.606492,-6.151760,7.452993,3.546235,9.757858,4.343822,8.307187,9.394370,-8.391786,-5.051581,3.013811,2.408324,-9.291384,5.677942,-9.866550,-3.574504,7.179243,6.219990,-8.127573,3.166141,-4.917363,0.565476,-1.033141,-0.544070,-1.809759,-2.489105], dtype = "float64")#candidate|4946|(520,)|const|float64
call_4945 = relay.TupleGetItem(func_1497_call(relay.reshape(const_4946.astype('float64'), [8, 5, 13])), 0)
call_4947 = relay.TupleGetItem(func_1499_call(relay.reshape(const_4946.astype('float64'), [8, 5, 13])), 0)
func_417_call = mod.get_global_var('func_417')
func_419_call = mutated_mod.get_global_var('func_419')
call_4948 = func_417_call(relay.reshape(var_4929.astype('float64'), [8, 10, 16]))
call_4949 = func_417_call(relay.reshape(var_4929.astype('float64'), [8, 10, 16]))
func_2016_call = mod.get_global_var('func_2016')
func_2019_call = mutated_mod.get_global_var('func_2019')
var_4955 = relay.var("var_4955", dtype = "float32", shape = (9, 220))#candidate|4955|(9, 220)|var|float32
call_4954 = relay.TupleGetItem(func_2016_call(relay.reshape(var_4955.astype('float32'), [11, 12, 15])), 0)
call_4956 = relay.TupleGetItem(func_2019_call(relay.reshape(var_4955.astype('float32'), [11, 12, 15])), 0)
output = relay.Tuple([call_4923,call_4927,var_4928,var_4929,var_4930,call_4932,var_4933,call_4945,const_4946,call_4948,call_4954,var_4955,])
output2 = relay.Tuple([call_4924,call_4931,var_4928,var_4929,var_4930,call_4934,var_4933,call_4947,const_4946,call_4949,call_4956,var_4955,])
func_4957 = relay.Function([var_4928,var_4929,var_4930,var_4933,var_4955,], output)
mod['func_4957'] = func_4957
mod = relay.transform.InferType()(mod)
mutated_mod['func_4957'] = func_4957
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4957_call = mutated_mod.get_global_var('func_4957')
var_4959 = relay.var("var_4959", dtype = "float32", shape = (792,))#candidate|4959|(792,)|var|float32
var_4960 = relay.var("var_4960", dtype = "float64", shape = (2, 640))#candidate|4960|(2, 640)|var|float64
var_4961 = relay.var("var_4961", dtype = "uint16", shape = (336,))#candidate|4961|(336,)|var|uint16
var_4962 = relay.var("var_4962", dtype = "float64", shape = (36,))#candidate|4962|(36,)|var|float64
var_4963 = relay.var("var_4963", dtype = "float32", shape = (9, 220))#candidate|4963|(9, 220)|var|float32
call_4958 = func_4957_call(var_4959,var_4960,var_4961,var_4962,var_4963,)
output = call_4958
func_4964 = relay.Function([var_4959,var_4960,var_4961,var_4962,var_4963,], output)
mutated_mod['func_4964'] = func_4964
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4889_call = mod.get_global_var('func_4889')
func_4891_call = mutated_mod.get_global_var('func_4891')
call_4978 = func_4889_call()
call_4979 = func_4889_call()
func_4957_call = mod.get_global_var('func_4957')
func_4964_call = mutated_mod.get_global_var('func_4964')
var_4989 = relay.var("var_4989", dtype = "float32", shape = (792,))#candidate|4989|(792,)|var|float32
const_4990 = relay.const([0.073780,5.478221,-0.603596,6.263555,8.363324,4.028496,6.241394,3.177920,-7.470613,6.943984,0.874616,-2.401715,-5.098304,-6.318839,3.578852,9.969577,0.793137,0.384435,-2.670159,-2.595858,-3.369627,2.426043,-1.883672,-5.239970,-8.408445,9.621769,0.552835,-7.517280,-6.357109,4.773539,-6.884139,-1.323716,-4.763633,7.736383,-0.574856,-4.911110,3.954137,-2.855653,-1.088883,9.202975,-0.168123,-7.933763,-5.245748,4.116517,9.872188,4.045224,3.131746,7.454559,0.039344,-3.612284,-1.277977,-4.835773,-0.045436,-4.715370,-6.346414,-3.453568,-7.998537,-8.648953,6.048690,4.551948,-6.990177,-4.420776,-1.979037,9.069488,8.660025,7.036951,-4.923902,6.628795,-5.969687,8.179960,8.340874,2.173509,0.740556,6.185034,0.574713,4.033353,0.438280,4.349565,-2.837881,4.970659,2.651397,9.564549,4.622595,-6.072630,-8.749483,-7.417150,2.517387,-0.963097,9.180565,2.258671,-7.884279,-8.162274,2.552499,-8.395192,7.749014,6.383317,-4.928168,-4.015315,8.146998,-6.526929,5.401584,-2.262487,-7.195028,0.006190,-6.208830,0.291806,7.507358,-3.046950,1.915412,-6.827958,-6.041666,9.830772,0.325686,-4.115626,4.028858,-8.991283,4.406441,4.516435,3.378886,9.603189,-2.469400,9.664943,5.780302,-5.170686,-3.042443,7.326045,-9.289329,7.156439,-2.068010,-8.917677,-6.954156,-6.025166,1.995212,-7.229742,-6.431694,-2.688608,-5.525226,5.143657,6.661982,-1.328205,-6.700985,7.728059,5.244639,-9.368974,-3.140720,9.705561,-0.662019,1.789906,5.402615,-2.064228,1.150018,-5.324759,-7.457267,4.617483,4.753303,5.800162,7.211178,-1.218269,-9.022337,0.955179,-0.013787,6.786019,-6.938283,-9.583341,-1.238186,-0.063670,-4.913179,1.321675,-5.657347,-1.746671,8.773856,0.501976,-9.607640,9.160386,9.271421,-0.726283,8.258854,-5.294844,-5.845790,1.737962,8.396194,-2.944050,0.554716,6.509819,2.707831,3.378755,0.687514,8.342308,0.537455,-1.953654,3.854722,-3.722933,8.211880,6.320294,2.133777,-8.428719,-0.012075,-9.432219,0.345097,1.161359,-4.361257,0.269811,7.353889,4.286311,2.499043,2.810589,-5.148807,6.471654,-4.694743,1.321648,8.908826,3.504921,1.855048,8.360442,7.518099,5.951570,-5.745291,2.718074,6.247089,-5.679087,8.827997,-7.121414,2.089973,4.292142,-8.172846,3.980916,-1.660750,5.325804,5.084504,-7.407625,7.865566,-1.388726,-6.875830,9.685155,-0.652162,-5.718459,-5.961831,-3.977887,-6.963525,5.805157,-2.257790,4.717272,1.224737,-5.001274,7.502566,2.928631,-3.280937,-1.844156,-5.977061,7.946842,9.611837,-5.767383,-4.934871,1.019636,-5.689145,0.584614,-5.242555,5.962302,7.270800,-9.789101,1.591934,0.517423,8.569036,-0.043934,3.177756,-2.174447,-3.795589,-4.627959,7.180251,8.572099,8.484426,3.470556,7.907714,8.868083,-7.532584,-7.840475,-5.431936,-6.272972,8.614403,8.442697,-7.864040,-7.721048,-4.141669,4.410868,-5.176507,9.858725,2.551059,1.773671,5.355481,6.075220,-8.273784,9.116331,8.054434,-1.708394,4.879824,2.168612,-2.729647,-3.967146,8.260028,-1.313741,-2.731357,8.495987,-1.930301,2.094417,4.869361,-6.456508,-2.772958,-5.959555,3.604147,-5.290283,-1.422667,-3.813842,5.517219,1.471949,-3.680638,6.876812,6.736935,-1.336298,7.867816,-0.088649,-3.244559,4.320692,-1.371392,-8.072047,-5.355548,-2.811780,-6.750109,1.881948,2.294247,-2.400325,6.856842,4.658841,-2.221464,-3.966991,3.488734,1.015885,7.138216,-0.578413,-7.980045,3.321083,7.711089,-5.414532,-9.826465,-5.939718,-2.827267,7.148713,-3.668377,2.475373,-1.383628,-1.691518,-9.514049,3.645967,-4.586351,9.270844,-6.742073,-8.918749,0.852533,-3.430207,-1.562609,2.585173,5.526393,-1.944325,0.388086,6.519103,-6.063449,3.889526,8.290025,-5.033707,-1.941696,-8.559822,2.675687,0.735453,9.629300,-4.797782,-8.436288,2.544699,5.216938,8.507880,6.888925,-0.537952,-7.561299,-9.256643,2.955568,-4.931267,-4.125455,-5.220524,3.430674,-5.612857,3.643792,-5.157306,3.008678,-4.895131,-4.711716,9.297027,-7.542907,1.778377,-6.321394,-0.719717,-0.152884,5.157630,4.015114,-4.362046,1.491692,-4.045656,-0.840452,-3.833765,-7.640960,-5.326053,-4.791310,-9.368425,4.789234,-8.387805,2.410097,5.153458,4.314848,-3.513825,1.777880,0.663097,-2.481949,4.486163,-0.256554,-1.583782,-5.978814,2.768806,-7.493427,0.119443,5.216067,5.123139,-0.513518,0.956547,-2.529019,0.338618,-7.037216,-0.615761,-0.941104,9.121310,-7.145816,6.812376,-8.078761,4.114422,-5.044750,2.740933,-2.388144,-0.917949,1.199037,-3.562590,1.404057,5.498157,-2.168021,-6.110447,-4.402757,-8.240174,1.253895,1.055441,-5.818951,1.119094,-3.767495,0.099184,9.810993,3.210611,5.432060,-0.139546,2.343471,-7.972517,-7.955654,-4.655133,5.016732,6.848105,-7.317642,-2.888195,4.513852,0.078689,-3.230420,1.479258,-3.553065,8.977884,-0.265399,2.953914,2.917075,-3.294431,-2.964147,-9.776073,5.791137,9.897049,0.950162,-7.311362,4.086537,-7.739092,-8.758909,4.721629,-7.622588,-9.615045,4.759823,9.806870,-4.338112,-3.797511,-3.979376,-3.805536,-1.984704,-3.730322,8.757998,-6.571961,-4.839125,0.238944,-1.063039,-0.866413,0.017206,-9.449331,-9.956383,1.607487,4.952826,4.030232,1.926449,7.683072,9.964551,5.988026,-1.074909,6.203952,-3.787661,-5.587017,-2.160978,1.779968,5.426387,-9.886850,1.206992,-2.372172,4.232470,8.868091,2.626695,-6.609812,-6.671326,-9.897173,3.248007,9.675781,-3.825745,7.737587,-2.306687,-0.668460,-7.373191,2.565156,-6.362413,-3.403252,-7.341701,-2.637156,-1.279048,-5.016905,-7.453740,6.432191,8.624662,5.921706,-6.515535,-0.554232,-1.216172,5.588852,3.411632,-7.293413,-6.159361,3.007118,-9.526538,1.950365,-2.527440,3.404478,-0.042072,-9.279733,-5.755067,1.670455,-4.695573,2.866280,-1.204925,8.342591,-9.603167,2.415086,-4.369034,1.933347,9.029623,5.879197,-6.802831,-3.285516,-1.931266,-1.939618,5.178199,1.347355,9.548723,-8.460328,8.318280,4.447207,1.437263,0.822530,8.110001,2.378083,2.913916,-1.833936,1.958883,-6.177508,-3.416662,-0.040467,-9.159545,8.245512,9.586308,9.645011,-5.924105,-1.660016,1.094182,6.010090,-2.890687,6.957871,-0.938271,5.662914,9.785458,0.760018,1.574968,9.114933,2.937085,6.631971,-6.074736,-7.965516,-8.641974,-0.772039,-0.243176,-9.118200,-9.774557,-7.882780,-4.028446,9.175394,2.873697,3.699120,-2.290524,-4.040999,3.272573,-6.159920,-2.005050,-7.021841,7.445847,-9.238092,3.049933,-3.814362,0.826660,-5.197901,7.305373,4.923080,-9.918573,3.944276,9.776675,-0.927547,4.141767,-4.669136,1.693399,-0.638248,2.745623,-7.532793,-6.144504,8.363626,9.924700,2.452578,3.697877,8.099254,2.868095,-1.682692,-9.139560,7.213606,-1.184057,4.872270,5.471442,-0.470734,7.110253,1.276124,-6.725938,7.448910,-4.304594,-7.028759,5.785343,4.958124,-2.538437,7.717070,-7.074651,-7.340220,-3.242897,2.878512,-5.035922,-4.909543,-2.894074,0.479914,-9.309458,7.310616,4.792134,-5.324430,8.206565,7.245080,-1.375141,6.145541,8.857450,5.352460,9.217640,0.758704,-6.669541,4.882679,1.784159,0.945228,3.583362,-8.214819,-9.587053,0.792769,-3.395539,9.667758,0.210954,-2.502589,5.087878,-7.271656,-3.946674,-6.791288,-0.913716,6.811215,-6.863298,4.229540,8.570661,-9.767744,7.961015,-8.409730,-0.387651,-3.482500,-6.247503,-0.500884,-9.239450,-6.275023,2.256946,1.127982,8.756902,9.293104,-0.383031,0.692997,0.754138,6.010244,-1.534859,-1.797856,-8.816611,-9.021995,4.390750,-6.585496,-4.722642,-1.439973,7.994313,9.164463,0.653545,-1.211630,5.344257,2.603740,9.416328,6.985004,-2.858947,-3.796316,-7.771830,5.486336,8.102913,9.524517,6.151999,5.876271,-2.580039,6.342136,-0.554343,2.425794,7.452903,-6.597106,3.657170,-1.901141,5.209440,-5.977864,-6.503758,-1.634751,6.243608,0.916812,6.356315,7.246118,8.028429,4.609794,-2.136277,3.004018,-8.166985,-1.229519,-5.710104,2.364951,6.302411,2.279768,-7.720263,-1.881781,-7.349495,-2.899341,-1.281920,9.079533,-9.412078,4.624918,9.105924,-5.192625,-4.228848,8.865150,-1.579229,4.261137,0.020765,8.962776,-5.386489,2.921806,1.418052,-9.328896,-4.928997,8.050486,7.383427,1.601503,1.240020,-1.998095,1.674706,-9.598077,-4.053156,2.114444,2.039179,7.408491,-1.231781,-6.132801,-8.793424,-2.001033,-9.896261,1.445992,-9.759751,-0.662983,5.700535,-7.172268,-6.140555,-3.802237,-0.690070,-4.760276,-1.866619,7.814137,4.790770,-3.739292,-7.931775,-4.842976,-5.755589,-9.222200,-7.318532,-3.245022,7.094766,-1.885301,-6.695816,2.127487,7.137736,-9.616938,0.726834,-1.527799,9.031515,4.427758,7.391775,-6.149407,4.707346,8.443091,5.507646,2.764086,-2.234066,-4.731522,-7.077405,4.044458,-2.726627,9.070443,4.094174,-7.852465,-7.694218,-6.004725,6.058103,9.095268,-9.132114,-0.222133,4.794352,-5.102066,-0.587715,-8.536737,-6.813004,-4.677884,4.007708,1.989357,0.429777,8.337876,0.461232,-9.353995,-6.434626,7.101743,-2.513330,-3.798621,-1.874423,9.544512,2.692690,4.379583,3.201350,-7.614088,-8.554574,6.144833,7.231523,9.107110,-0.146616,3.009261,-3.106364,9.728993,5.493959,-3.150973,9.211875,-9.492193,4.786030,-7.533348,2.905289,5.219419,-2.832384,2.682908,8.401886,-7.672547,-4.088163,-8.401925,8.538735,9.204748,3.898971,-8.222941,5.496741,7.914515,-4.426387,0.516864,3.818267,5.242677,8.507833,5.301070,-7.151657,9.038612,-6.252474,4.575873,7.512486,-5.260957,-5.350313,4.284210,0.190584,-7.904257,3.405399,7.110056,-2.778608,-3.021174,9.341003,-7.764033,-1.314744,4.477772,1.121389,-9.964945,2.275596,1.936459,3.653577,-0.030383,7.054545,-9.101661,-6.857498,8.435627,3.221940,5.064001,-2.556624,-3.998978,5.085443,1.924951,-9.087478,9.444121,-2.040394,3.282769,-7.528041,-1.851204,-2.682621,-5.145640,-2.454569,3.249248,3.650980,-2.552788,-5.216853,-4.482684,-0.695485,-0.837217,5.728249,2.672072,-5.675990,3.247210,-1.604576,-1.913145,6.016448,4.165891,4.863122,5.726626,2.946410,-6.958736,-8.166854,4.744290,-9.253939,2.964871,-5.537230,4.101513,3.104206,9.884732,-0.023410,-7.758009,-7.520931,3.937801,-5.298861,-0.578699,-6.499074,9.471133,-3.005657,-7.273303,-9.206289,-9.169969,-9.113097,2.785428,4.043552,9.778200,-9.202215,9.321047,2.584264,-9.625214,7.189682,-8.847693,5.877870,-4.191796,-4.256013,-8.301478,-9.612935,5.992636,-0.778488,-1.664467,1.679872,0.329005,2.954145,-1.322800,8.408901,-1.155990,-2.840291,6.427536,-3.115794,3.330039,-4.148275,6.139641,-1.038346,2.549987,8.498812,-2.622712,5.413463,-5.852895,-7.554421,4.239104,-4.094369,-0.165260,-3.454956,-5.076595,7.238450,-6.287021,-2.785155,-7.554571,-0.105017,1.596007,5.494055,7.548059,-7.799806,1.638832,-6.880226,-4.994356,-0.499855,1.816319,-2.307680,3.327836,9.570819,9.027037,6.486932,9.439537,4.270836,1.451564,-9.087962,6.998182,5.461489,-7.049004,-2.548132,-4.075998,0.814325,2.506377,1.864960,-6.873162,-0.374905,-7.742859,9.450528,-1.651672,1.346471,-8.688367,-7.594207,-1.662077,-0.757615,9.378373,9.221136,-0.897598,-5.485314,8.870740,-1.693936,-4.396408,4.401197,9.190003,-6.642372,-9.733496,6.675185,-2.280803,-1.300637,1.934239,4.607924,5.533195,-3.806396,1.080230,-4.739795,8.795011,-6.145625,9.064489,-5.106992,0.411059,-0.925857,-0.525125,-4.205907,6.641849,5.095965,9.262426,-1.064139,-5.426971,-7.614533,5.089842,-7.339859,-6.673331,-8.867341,-1.759804,-3.915980,-7.613368,8.423785,-2.896432,7.377524,8.070647,0.874227,1.652857,3.237886,-2.667193,5.362885,-9.207532,-8.743324,7.711235,1.902617,-3.830971,7.872133,-3.579344,1.214586,-5.110762,7.616163,-6.066664,-9.102852,-9.078360,5.515540,-4.775041,-3.092736,-7.856503,0.589552,2.029396,0.071525,-9.267808,-4.095217,-7.726199,4.394836,-6.592378,-6.454322,1.478387,-7.714513,-2.790933,6.684081,0.855185,-9.997499,-1.359737,6.514927,-3.165488,5.610659,5.009529,0.007816,-7.586913,-4.806071,6.991852,6.904369,1.106360,-3.615464,6.502788,-5.707823,3.875443,-6.511116,-3.377326,-7.294968,-2.409567,0.936684,8.912282,9.275798,-9.671152,0.039413,-5.148628,-5.049616,-8.906092,-8.981230,-8.708752,-2.510711,9.410177,-3.974328,-7.381585,-2.375030,-1.395591,-1.392728,-4.709499,5.535991,5.899274,7.941917,9.561248,-7.210940,-0.666218,2.081503,1.156005,4.558791,-4.339434,3.040292,-0.229744,-8.084052,6.899004,-2.913248,-4.039448,-2.366964,-8.266926,-7.008469,-2.898444,-4.648813,5.864758,3.288732,5.378392,8.542450,-2.550400,3.760434,-0.752269,-0.158330,-9.456676,2.408672,-2.356997,2.433350,8.310437,4.500654,-4.985559,4.402906,6.199132,-1.689666,5.301427,0.316048,-4.066957,-9.050215,-1.420666,6.372681,0.129699,5.322623,-6.523688,0.478891,-6.664660,7.389956,-1.706248,-8.629558,-7.593065,-4.161413,-6.724734,1.128422,-4.162810,9.882375,-4.355009,-1.605478,-6.153718,-3.256266,-5.300585,-4.846242,-6.737850,4.969610,-6.350326,-6.646042,-6.831217,-5.507345,-8.805364], dtype = "float64")#candidate|4990|(1280,)|const|float64
var_4991 = relay.var("var_4991", dtype = "uint16", shape = (336,))#candidate|4991|(336,)|var|uint16
const_4992 = relay.const([-9.920794,-6.810763,3.132873,-2.670536,4.841138,-6.851348,5.143814,8.208256,6.307435,-7.452678,-8.940987,1.874246,-5.547062,8.159268,7.813412,9.767501,-8.023876,-4.309808,-3.564430,2.508455,6.193462,-8.693050,2.864172,-3.463455,4.467246,-0.393611,-8.877449,-7.320568,5.123872,-3.974763,7.224434,-9.703541,3.587586,-6.111490,3.438483,-7.884638], dtype = "float64")#candidate|4992|(36,)|const|float64
const_4993 = relay.const([-7.230055,9.565963,-4.150680,1.623053,9.673814,7.899063,3.026804,7.796561,4.831576,-5.334228,-7.866420,-3.858681,9.483745,-3.787754,4.507778,6.476753,-6.238355,-8.902560,-1.983981,7.967461,8.648784,-4.846558,-1.294157,-7.841531,3.120751,5.615030,0.098038,1.448944,-3.350179,1.841179,8.320096,2.130207,8.586703,5.896886,8.575144,6.248985,4.333097,-2.983231,-3.455032,-4.545142,9.007953,5.444493,1.410313,9.807077,5.106065,-2.642335,0.930518,-0.604681,9.505887,2.553274,-5.067378,7.918049,-2.427816,8.593230,-0.843950,8.251842,0.673676,-0.610893,0.633173,-0.109617,-7.717330,-2.765890,-1.315271,-0.133819,-7.882264,5.488050,6.359070,-3.544576,-7.076552,-4.574257,9.241037,-8.942595,-0.543769,-1.353986,0.946562,-3.243839,2.741140,6.919438,0.267194,8.917678,0.077421,-2.334722,6.610086,8.434521,4.108124,3.731237,6.934582,9.255376,-5.041508,8.678567,9.795702,-1.850228,-0.641903,-0.458972,4.104282,0.985568,-1.865578,7.553829,4.002472,0.247423,-9.206985,3.221111,4.042795,-5.217949,-0.665429,7.166723,3.296049,-2.596119,9.928354,-2.309206,-6.422679,-7.699066,7.024826,1.677188,-1.965920,0.693086,5.181008,-8.415130,-8.208708,-7.792618,-5.912805,-4.543172,0.841013,-4.267489,5.179209,-5.103234,-8.392052,6.541611,-7.742479,-5.462106,6.888158,-1.801283,8.709859,6.349770,9.441811,-5.376860,9.788961,4.355563,-2.376646,-8.231493,-7.023515,2.303170,0.981833,4.519645,4.522338,-0.331245,1.695123,7.715412,6.746982,-1.729687,1.364010,-1.890437,4.019317,7.636830,4.218763,-5.004119,-6.652069,6.438246,-8.572494,-5.562810,-5.320260,-0.983215,4.252117,-4.028840,9.953309,5.353746,-1.518806,9.973643,1.644339,7.058048,-7.537432,1.400874,1.282111,-6.568317,6.995774,-8.215021,5.297290,4.594286,-0.632905,0.470400,8.510181,-1.954389,5.052181,4.949706,3.567156,1.530492,-5.860186,-7.205609,2.205113,4.918254,0.893964,4.644706,0.553085,7.355663,-1.595986,9.112936,1.374821,7.424859,5.530090,-5.145436,9.490867,1.868254,-5.032625,-9.456750,-6.759298,-1.254885,5.040635,8.017855,-6.499266,-6.196164,0.468251,-5.061294,-0.835907,5.672951,2.954042,-6.443424,-7.308696,-7.786253,3.439485,-7.737298,5.371529,-6.860667,-4.764593,9.386993,-8.445760,-3.971753,1.432299,5.156001,-6.495705,1.100726,-4.154440,7.260653,7.110848,9.862998,-9.903224,7.324299,-2.289517,-7.176617,4.734247,1.884073,0.409367,9.569322,2.421626,-0.238199,-7.364661,3.051777,0.046071,3.628302,-6.911229,-0.799123,-4.149082,-7.803410,2.183903,-0.066640,-7.750045,-2.635022,4.068500,-0.167112,-6.346730,-2.561581,1.346506,4.685889,-9.466730,4.644921,3.861770,1.558289,7.947833,-1.768525,-9.145731,5.735633,2.040193,6.011262,-4.960434,3.123957,-5.586555,6.774744,6.291825,4.445198,4.629382,-6.476596,-2.364331,-6.679771,-1.152504,-5.149545,8.135253,1.164804,8.536075,6.196884,0.977883,7.619758,1.660332,6.841005,9.130890,-7.871465,6.523959,-1.203365,7.946434,-5.106074,4.982162,-5.168751,-9.031409,1.887676,-7.261308,1.122402,8.126838,-8.297809,7.670170,3.393233,-5.997204,1.827638,2.854410,1.857539,3.701924,-9.093876,-2.525834,4.152314,-1.051730,-7.239675,-9.310327,-2.980517,-9.826115,-1.932416,-8.756423,3.710490,4.797605,8.764962,-3.198993,5.950809,-5.479516,-2.944071,5.327141,8.299880,0.140573,-2.849925,-4.321604,3.940509,9.346074,-4.194827,-8.916547,-2.457556,7.772796,-6.217364,-4.479232,2.051496,4.420550,0.752988,4.018075,-5.731297,1.028195,-2.320850,-9.272224,-1.731209,5.884121,5.904996,-7.650660,-6.652754,0.210208,5.225367,-4.545695,-2.721677,1.550507,-1.297381,7.639919,8.193048,-0.977241,-1.046300,7.077125,-0.285901,3.635758,-7.273708,-1.386597,3.014543,5.952132,8.423064,0.331729,1.172474,-7.933752,-4.961580,1.106772,-0.312140,6.864961,-2.152636,8.725487,-5.366623,-4.443448,4.547514,5.662446,7.933188,1.460752,-9.515427,-5.437777,0.904473,-6.445383,-6.582456,3.064281,0.029187,6.154843,-3.634250,-2.420732,-5.828216,0.268074,7.908362,-0.305766,-2.906009,-8.017433,0.454117,-3.960210,3.435882,-7.289897,-3.004650,-9.940508,-2.499078,7.064739,-4.883142,7.922390,7.361306,-1.515167,6.404085,8.821239,1.189189,1.763537,-0.958380,4.410546,-8.886293,-0.053468,-5.025872,-6.590531,-1.018595,7.299064,2.906497,6.976771,6.350851,-1.620274,3.102058,-0.958096,1.575630,1.133406,-5.503952,9.027280,-0.914657,1.569869,-8.039338,6.548060,-6.140930,4.098753,-4.634238,-6.091431,2.052838,-7.491263,-1.036251,7.001702,-0.269123,1.352378,1.842874,6.504028,-2.513162,-7.153859,-4.311162,-4.808828,-4.615360,3.961856,-7.221431,7.296723,-0.459598,-7.143659,-8.573870,-8.274795,-2.115654,9.141208,-6.472891,2.811664,-7.819428,5.235951,-9.108025,9.592384,0.227296,8.351400,-6.314268,2.906494,-0.439531,7.178137,4.724330,-9.023391,7.343534,-1.740519,-4.816892,4.516585,1.425551,-8.145299,-2.442486,6.350513,-0.671796,2.325852,4.977781,7.164141,3.746420,-7.792901,-7.458731,-0.455842,2.040601,8.648414,7.211231,-3.026962,2.232876,-3.263391,-1.499202,2.124760,-6.731978,0.686318,-7.716215,5.111138,8.629933,-5.318222,-7.452922,0.723771,3.874790,-6.560582,8.845801,-3.626759,0.339885,-3.680596,-7.611643,-0.039193,8.244632,-3.617234,-1.625851,0.504143,6.127824,5.798718,-0.204726,5.914052,1.967908,-0.350439,-7.845111,5.836038,3.929936,8.404998,6.967945,-3.311274,4.209506,-4.950889,0.172559,1.725488,-7.437779,-1.095717,-3.680710,-9.555620,1.990450,-8.821235,7.329188,8.262548,-1.234189,-6.723597,-7.322952,-6.375404,5.267257,7.046331,-0.216860,-7.264689,2.722217,9.833023,-2.939253,0.951982,1.640583,-9.565288,2.925475,8.458182,-0.635597,-1.240474,0.400909,0.600389,0.763840,7.653550,0.746052,-8.279403,-3.008492,-8.742697,-2.097123,5.524791,-3.569348,-6.413212,2.780198,3.999391,-2.647058,0.879900,-2.242152,-3.420630,9.821323,0.847847,5.979683,-6.372584,2.177474,9.895032,-0.717228,2.393938,-3.273118,9.175420,-9.432663,2.107122,-1.470158,-2.058569,9.300420,-1.978879,-1.952784,-5.530005,-4.978239,5.029509,5.074281,9.485980,6.111264,-8.394289,-6.597693,-8.530142,-6.401720,1.573951,-4.480032,6.158970,1.391597,-6.125366,2.973704,0.183236,1.860615,-7.544774,0.302429,6.563703,-9.456051,-8.892286,8.186159,8.112656,3.817216,-6.652213,-8.063868,2.765083,0.066504,3.755306,-9.673896,-3.583783,-4.204538,5.379310,-2.038405,-3.078108,-4.026406,-2.323131,1.342857,-9.444851,-5.947218,-6.164366,-1.291118,-8.472047,-3.750623,3.259474,5.258358,-9.373425,-6.891303,7.738957,-8.649377,8.662460,8.802945,-2.989542,-5.657498,0.482350,0.977473,7.867812,3.743238,-2.048930,-0.046235,0.037467,5.990020,0.833612,1.559171,2.781241,-0.890574,-8.670722,-9.109250,4.402750,7.170553,8.885843,-7.712749,6.155127,3.033303,5.446627,4.931460,7.999393,3.882581,-0.100698,4.620019,-7.454672,-0.177075,2.791596,-7.713275,7.108771,6.874585,-8.466753,-0.470349,-8.333792,0.984461,3.783066,-5.086414,-0.303218,2.616272,-4.891463,-1.371294,-0.057001,3.858736,5.821021,-9.104988,-6.059258,4.336975,-8.236474,-1.284594,1.740262,-1.352969,-3.723822,7.133645,-6.568801,7.674694,-9.023918,-4.546109,-3.794977,-7.398888,0.334123,2.022057,5.824684,-8.202717,-9.594917,-8.222392,7.310801,-1.668213,6.776454,-2.275986,-7.887428,-8.410447,4.976840,6.624879,-3.770270,8.509530,7.205419,3.557926,1.212659,-8.225693,-9.889432,-2.856359,4.146320,9.038212,-3.226613,4.460322,5.022208,4.709264,-1.304872,4.391103,2.202598,-3.272753,6.086331,4.078409,8.513719,-8.749615,-5.150799,9.249306,7.997773,-6.108465,-7.706248,-1.905621,3.996793,-4.822783,-4.577719,-8.968900,-1.825397,-3.292196,8.368068,6.369036,5.538107,-0.906114,-4.467849,-5.193996,6.353174,4.954448,-9.872343,5.335033,-2.138133,9.817478,3.465454,-6.269382,-2.822778,8.312879,-9.823131,-0.861520,-7.476347,1.136233,9.378657,-9.220018,-3.106650,-9.329695,4.683346,-0.215401,-4.796044,6.545059,2.608197,7.129821,-0.420318,4.452578,8.080474,8.378244,7.975830,-2.775822,-8.682705,9.725055,7.077298,3.753195,2.658955,-1.074884,0.935275,-0.805442,5.257623,-5.057415,-6.871237,-2.216950,1.351280,-8.093041,-9.093344,2.937152,1.116694,2.211474,3.281547,-8.736472,-5.198956,-8.406564,-7.543335,-4.366528,-5.285440,0.071623,0.859747,3.886375,9.708755,7.530261,-9.800061,-1.406008,4.486201,-8.679720,-7.382263,5.296783,7.203709,-9.795476,-6.592596,8.636605,8.338882,-5.940508,-0.985480,-7.904521,5.533577,6.792108,-2.767259,2.005872,7.244761,-4.336291,-0.863882,-6.187058,7.377678,2.423962,-7.905005,7.707435,-9.900719,-3.084935,4.791572,-4.132008,-5.490930,-7.075655,-0.822432,-0.514569,-5.278833,1.786672,-9.714858,-1.251772,9.181145,-2.268326,4.314109,4.463445,8.158722,-0.392900,-9.529447,1.148944,2.306041,-8.202353,2.370233,8.996604,0.307199,2.869639,3.197673,-9.522119,7.605150,-6.904694,-1.873183,-1.214389,0.471144,-8.818075,-6.324617,-3.138014,7.382453,8.532611,6.889488,-5.159739,4.979568,-8.828949,-4.225992,3.607732,1.743657,-5.884039,-5.882737,1.013444,7.932818,2.177252,8.794006,1.165966,-7.389066,-5.933233,-8.930835,2.027970,-8.173997,-8.105020,-8.955070,-7.781975,3.888938,1.491116,-4.663740,8.242921,-3.294592,0.526477,-3.576096,-5.593468,5.833335,-2.239339,-1.721832,7.291010,9.519616,3.582497,-7.357385,-4.371105,-3.660179,2.456510,-8.338269,-0.445128,0.670179,1.555347,9.397396,5.592719,-1.051389,-1.452279,3.912694,9.946317,5.097261,-5.434571,-1.656271,6.028996,-6.538243,9.081632,6.365572,2.500992,0.454635,-5.870027,-5.259147,3.862133,8.938375,8.702060,-4.419937,3.622044,2.922467,4.419487,-0.674067,-8.679217,5.227168,-6.492798,-2.661619,4.837650,-1.723549,-4.953953,0.094734,5.910790,5.307564,6.378034,2.834703,5.104974,-2.025548,-8.272733,6.121730,8.398153,6.205551,4.314832,3.812882,6.260551,-0.279271,9.103672,-1.678879,-5.694769,8.820466,1.519810,-1.605845,-9.864496,7.584583,-1.563888,5.065995,3.770519,6.804439,-3.548857,2.676177,-5.068379,5.738491,-7.044263,-5.463723,8.130802,-8.954040,-8.083850,5.717294,-4.742717,3.183840,8.784530,-1.002483,8.338824,2.969285,3.203644,-5.083903,-6.943410,2.287266,0.396970,1.041721,5.114838,3.276806,9.434726,-5.751946,0.910707,-1.432667,9.675539,6.017579,-6.088168,-0.986355,3.999815,1.910357,7.898663,-3.559812,-6.742319,7.484541,-6.625454,8.929310,-7.347755,0.174277,5.076274,-2.582776,-1.751427,-1.834172,4.092617,-1.743518,2.425787,-6.724582,-5.271745,3.550964,8.850857,4.030589,7.022747,3.832027,-4.983581,2.572685,-1.044898,-9.435281,3.158920,3.357536,-6.938265,5.977569,6.848463,-9.795038,1.285980,-9.675485,-3.456655,-7.636564,-5.280728,0.267924,-8.838829,-1.283049,3.606617,8.745347,4.316390,9.428950,7.698073,0.581002,-2.221579,-6.704026,5.543261,-2.827134,-1.061059,6.153782,-8.690165,-3.532069,7.713955,3.770541,5.817167,3.601552,0.851972,-5.235524,-0.598232,2.692151,5.460193,-0.858567,-0.177429,9.274073,5.397427,2.391654,3.197064,-7.173306,4.385768,9.676327,-3.595163,6.930450,5.513177,-9.574986,-8.869142,8.210443,-2.957397,-7.119112,-2.596532,5.184382,5.306505,-6.042003,-6.159820,5.075858,-3.831468,5.419868,-7.410763,0.274464,-0.740007,4.293510,6.074165,6.892800,-6.009366,5.855910,8.171137,-6.629589,2.805321,4.274116,2.293219,4.941806,-4.610803,-9.153045,-1.652277,-6.563870,-0.075543,3.822653,-3.286709,8.425814,-9.643931,-3.730642,9.092428,-0.243669,2.310733,6.138371,1.004552,-9.478866,4.104804,-0.511954,-0.510230,-6.688565,-1.003569,0.875810,-5.200211,7.408699,2.837968,-9.787148,-1.387670,9.074502,0.419619,-8.351044,-6.788344,-0.901096,6.712950,7.461933,9.955566,-8.435039,5.582123,4.126946,-1.179544,-2.251927,-7.087699,0.049256,-5.215931,-8.886536,6.226686,-6.322203,5.906176,5.054733,6.997217,4.267841,-1.883567,-4.004054,4.283573,-3.622202,-8.677765,1.058968,4.451482,0.610625,1.276787,6.700219,-0.322337,-7.471597,-7.640263,-8.752134,-8.752502,5.278672,-4.758625,-0.136865,1.275314,3.866820,-5.641413,-0.497087,0.392004,5.585165,5.823528,0.023497,-6.751662,3.657201,7.706611,-6.446982,8.561300,2.826551,7.548795,6.745770,-2.423943,-6.631507,3.382452,-2.860466,-9.984750,-6.752121,7.097942,1.309302,8.196492,-3.108148,-6.358225,9.733751,-5.796637,0.566220,-7.756623,-9.561296,-1.647122,0.294744,0.878170,-5.804459,8.939938,8.015556,8.233834,-6.152464,-2.086129,6.904904,2.196347,-3.662458,4.911447,5.653457,2.377160,-2.149752,-3.807282,3.508846,-2.742378,7.083644,-6.907940,-0.454139,1.535935,1.274218,1.801073,2.370739,2.412790,0.275193,6.259593,-2.186475,-2.427886,-8.487335,4.974167,5.202517,-0.690705,9.867418,2.135009,-4.732573,5.053121,-3.078348,-9.629113,0.237152,-0.174200,2.962189,-0.478025,-1.796291,-3.949459,2.298478,-7.033946,0.614135,-6.988348,-0.936434,-9.956081,-1.961487,-0.686717,5.913219,-0.867168,-7.082896,1.388644,5.998372,2.470980,-8.593082,-1.027970,1.115490,-8.558357,8.286698,-7.153135,6.278344,-0.372402,-4.859135,-7.885982,-9.331163,-8.356796,2.937142,-3.818513,3.792232,-7.431226,1.667960,-6.247637,-0.167868,-1.040298,5.201477,0.678470,9.133205,-3.502410,-7.809557,6.475704,-4.590519,0.033840,9.278159,-9.708358,9.768473,8.447098,2.890233,-4.781034,-6.349601,-0.258257,5.455485,-8.970987,3.026501,-9.201144,-8.848272,-2.092305,2.573523,-5.797278,9.544012,-2.965628,-6.226430,-6.761973,9.093320,6.881015,3.296613,9.534544,3.056758,0.278436,-7.259784,-0.278131,-1.788438,5.262034,-3.593512,-7.802001,-0.297425,-2.611536,-8.136780,9.620326,6.940074,-8.840939,-9.515204,2.838118,9.455921,-7.841096,-3.839085,-5.042694,-8.225269,2.282613,-6.748892,-3.309651,-1.647343,4.057832,4.173285,-2.287436,1.489326,1.825512,-4.129655,0.951175,-6.941885,4.887754,-3.639345,9.425367,-5.666512,7.544275,-7.399009,5.071906,-2.134461,-0.599034,3.583202,3.692748,9.663545,-5.482272,3.617738,9.930551,-0.056143,7.526964,0.413603,0.599609,5.434156,-2.602634,-5.026941,-3.168743,-0.767682,1.117466,1.001304,-2.791327,3.488174,-8.096767,-0.049837,-1.350882,-1.873717,3.498617,-5.876808,2.681788,9.624777,-7.245530,-9.693409,-3.121647,4.060334,-5.879649,-9.264393,1.914681,-3.091329,7.423257,6.253236,6.869976,1.184907,4.589148,9.151867,4.112540,-1.055654,-5.569111,-6.224049,0.588636,-4.761149,-7.024039,-7.921489,7.215859,4.580023,3.075960,-2.894996,-9.185984,3.795184,-4.345557,4.986236,2.412123,1.383895,-0.874489,4.992422,4.657412,7.263463,-8.532810,4.861089,-1.830580,-3.775904,9.601406,-2.948835,-3.179219,-6.056862,-1.333352,-0.127589,6.525411,-8.363085,9.491796,-4.485395,1.421984,-8.938616,3.513718,3.504912,-6.542883,-0.732901,1.684346,-9.859026,-5.455379,-6.662205,-4.951681,2.468861,-3.771584,4.840679,8.425498,-4.963981,9.802010,-1.965804,-3.123609,3.009381,-5.330203,-4.942295,-2.016151,9.841680,9.986929,-5.200943,7.419172,-6.401755,5.203017,1.526250,7.096932,5.599768,1.961423,6.824439,-5.928450,-6.088129,-5.585415,-2.654873,3.573461,-5.590919,2.169501,-4.451580,-8.785652,-3.871071,-1.663070,-3.759362,6.799556,1.492546,5.861021,-6.138273,2.792431,-8.160212,1.973334,-0.271913,4.545970,-0.088223,3.632006,4.419610,4.906632,4.139045,6.028431,6.361962,-3.110704,9.236543,9.570888,5.481743,7.121795,9.099986,-4.220236,7.030850,-7.763107,-0.425107,7.923941,7.533988,2.051694,-6.004730,-2.035230,1.387917,-0.832724,-5.151634,-2.174988,-3.044621,-9.375416,-6.463061,-4.972183,-5.608200,1.083559,-9.312373,-6.181602,7.152350,0.319148,-6.122394,8.196528,-6.413961,2.057531,-4.473425,-2.545307,4.276106,-8.726668,9.803060,5.591957,5.099508,-2.975616,6.363802,0.644201,-8.171190,1.317079,-2.071691,-8.206003,-3.598928,4.019704,0.632422,-1.768382,-1.806272,-8.978197,5.919173,8.264188,0.667452,-0.344204,-0.406239,0.282290,-3.923060,-2.733956,-9.808222,7.996703,-2.157001,6.918867,-9.933754,-9.383218,-5.360787,-8.944443,-7.110827,-0.816879,4.678581,2.010134,-5.200187,-4.222325,0.798753,9.635645,3.623901,-0.615757,9.078889,7.439211,8.443315,4.123430,7.946186,6.109696,8.083451,0.571958,7.491875,1.025960,9.751710,-1.117931,-7.649880,-5.947961,9.331057,7.217873,-8.704189,-7.562395,-1.774564,4.657614,-2.951088,6.681735,-3.421128,-8.531369,9.199262,1.823157,-6.504963,3.327918,3.174869,-8.538876,9.559268,-1.516249,9.223661,3.379304,-3.734074,5.902754,-5.728350,-7.419858,9.435445,-6.195185,-5.364394,4.717836,0.234857,4.050684,9.255170,9.011679,8.620199,-6.555047,8.538114,-0.114897,3.156438,-1.711446,7.718728,-3.424572,2.028989,5.350144,5.536291,-0.926133,-1.059694,3.094653,-7.497889,-3.729746,-3.272233,-9.422626,-9.881132,8.765545,-2.146889,6.646568,7.375376,-4.093138,-0.617770,-7.534645,-1.506538,7.122484,8.957408,-2.017928,1.391055,5.207315,-1.957283,4.904185,0.786174,-5.991748,-2.088497,3.113882,4.806193,8.456134,1.877355,3.970635,-8.579839,-6.881477,-1.536252,1.318955,3.173201,8.396282,-2.150034,3.248071,-9.649293,9.244280,3.201357,-0.607298,-9.425854,9.549526,5.349276,-5.207729,2.611241,9.349420,0.133529,5.434596,-5.888275,-5.716643,-8.949297,-9.160988,4.850093,9.744214,-6.312821,-4.801948,3.634740,-7.200833,-7.339704,3.063200,-8.277082,9.839387,9.909860,-9.319731,-8.524078,0.995011,-7.362753,2.610714,5.098843,1.925813,-2.591843,0.268349,3.626612,-2.295106,-2.640370,8.664396,5.768781,-5.915964,9.294125,-4.349430,-5.278237,-7.379002,5.107354,-8.937210,6.617081,-5.125908,3.170698,8.876322,7.978435,4.985187,1.169931,-1.076111,2.342898,-3.841594,-7.780768,-7.191797,-4.049224,4.307335,-2.767700,-0.406537,6.233101,5.671797,-9.671777,5.923049,9.691350,8.830308,8.367292,0.622148,4.770148,-6.916179,-0.084281,-4.412310,0.047897,-0.821882,2.859207,7.154476,6.818895,-6.875622,-0.189366,6.417282,-0.912120,-2.870322,6.701572,-6.663995,3.911564,-3.955845,-8.888091,-3.556543,-1.675980,8.435959,-6.172920,0.158749,-9.908855,5.209812,2.976311,-4.686362,-4.097120,-7.210118,-6.189710,-1.867715,6.180279,0.732589,8.454244,6.289072,-9.806096,0.060868,-9.002497,-3.663290,-8.371567,2.041597,-6.760242,7.783783,-4.431960,-1.452645,6.986504,3.199580,8.581110,-3.056567,-5.688949,1.763807,-1.649089,0.849984,4.452231,9.281218,-9.302770,3.590246,-8.350771,0.281410,-3.924422,6.963898,-9.804733,-1.213980,9.673998,-1.295392,2.167680,-1.635803,-6.130912,-5.469137,9.606702,-4.746355,-9.211511,8.552593,-5.011286,5.101838,-8.727276,-7.550360,2.300196,4.523193,8.866996,6.646781,4.729387,1.385194,8.218110,4.850403,1.589019,-0.114405,2.903442,1.044694,-4.664231,8.532604,6.062737,0.860337,-9.813049,-5.990944,-9.662227,-7.964688,2.431826,-1.480928,2.459337,-0.116841,6.176987,-2.798611,-1.017222,-4.890992,3.538697,2.473714,-7.110323,8.469665,7.523860,-9.617426,0.424420,5.603766,-5.598446,-5.044229,0.693478,8.621466,3.853177,-4.873650,9.056785,1.630916,-7.447224,2.970916,9.614046,9.781971,-3.945304,5.501893,7.881376,9.721927,-9.000995,4.394225,-4.973158,-3.629794,6.430696,1.492471,1.576680,9.919313,0.699793,-2.094681,0.525417,-6.097545,-1.540547,-4.773486,-5.162309,2.336896,8.051667,8.969258,6.466980,5.387117,-0.096518,8.881914,0.448350,8.903045,-7.019197,-4.293533,1.524023,7.299820,-3.236831,0.949598,9.176221,6.459658,-6.562883,-4.505745,-1.741020,-0.733290,-8.146749,-2.688017,-7.892815,8.588056,-7.923768,-3.079687,-7.750823,2.285487,0.043157,3.019446,6.232313,8.111587,6.456369,-5.432261,-1.359480,-6.862562,-0.577308,-4.520216,-6.761699,0.365553,4.683631,6.277277,-4.623859,-3.861729,9.273749,-7.254210,-2.537962,-1.294238,-4.036039,6.427765,4.336408,9.977244,7.259031,-3.271495,5.000805,-7.209177], dtype = "float32")#candidate|4993|(1980,)|const|float32
call_4988 = relay.TupleGetItem(func_4957_call(relay.reshape(var_4989.astype('float32'), [792,]), relay.reshape(const_4990.astype('float64'), [2, 640]), relay.reshape(var_4991.astype('uint16'), [336,]), relay.reshape(const_4992.astype('float64'), [36,]), relay.reshape(const_4993.astype('float32'), [9, 220]), ), 10)
call_4994 = relay.TupleGetItem(func_4964_call(relay.reshape(var_4989.astype('float32'), [792,]), relay.reshape(const_4990.astype('float64'), [2, 640]), relay.reshape(var_4991.astype('uint16'), [336,]), relay.reshape(const_4992.astype('float64'), [36,]), relay.reshape(const_4993.astype('float32'), [9, 220]), ), 10)
func_1747_call = mod.get_global_var('func_1747')
func_1749_call = mutated_mod.get_global_var('func_1749')
var_4996 = relay.var("var_4996", dtype = "int8", shape = (729,))#candidate|4996|(729,)|var|int8
call_4995 = relay.TupleGetItem(func_1747_call(relay.reshape(var_4996.astype('int8'), [9, 9, 9])), 2)
call_4997 = relay.TupleGetItem(func_1749_call(relay.reshape(var_4996.astype('int8'), [9, 9, 9])), 2)
output = relay.Tuple([call_4978,call_4988,var_4989,const_4990,var_4991,const_4992,const_4993,call_4995,var_4996,])
output2 = relay.Tuple([call_4979,call_4994,var_4989,const_4990,var_4991,const_4992,const_4993,call_4997,var_4996,])
func_5000 = relay.Function([var_4989,var_4991,var_4996,], output)
mod['func_5000'] = func_5000
mod = relay.transform.InferType()(mod)
mutated_mod['func_5000'] = func_5000
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5000_call = mutated_mod.get_global_var('func_5000')
var_5002 = relay.var("var_5002", dtype = "float32", shape = (792,))#candidate|5002|(792,)|var|float32
var_5003 = relay.var("var_5003", dtype = "uint16", shape = (336,))#candidate|5003|(336,)|var|uint16
var_5004 = relay.var("var_5004", dtype = "int8", shape = (729,))#candidate|5004|(729,)|var|int8
call_5001 = func_5000_call(var_5002,var_5003,var_5004,)
output = call_5001
func_5005 = relay.Function([var_5002,var_5003,var_5004,], output)
mutated_mod['func_5005'] = func_5005
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4857_call = mod.get_global_var('func_4857')
func_4858_call = mutated_mod.get_global_var('func_4858')
call_5054 = relay.TupleGetItem(func_4857_call(), 1)
call_5055 = relay.TupleGetItem(func_4858_call(), 1)
func_1937_call = mod.get_global_var('func_1937')
func_1939_call = mutated_mod.get_global_var('func_1939')
var_5057 = relay.var("var_5057", dtype = "float32", shape = (7, 6))#candidate|5057|(7, 6)|var|float32
call_5056 = relay.TupleGetItem(func_1937_call(relay.reshape(var_5057.astype('float32'), [6, 7, 1])), 0)
call_5058 = relay.TupleGetItem(func_1939_call(relay.reshape(var_5057.astype('float32'), [6, 7, 1])), 0)
output = relay.Tuple([call_5054,call_5056,var_5057,])
output2 = relay.Tuple([call_5055,call_5058,var_5057,])
func_5063 = relay.Function([var_5057,], output)
mod['func_5063'] = func_5063
mod = relay.transform.InferType()(mod)
var_5064 = relay.var("var_5064", dtype = "float32", shape = (7, 6))#candidate|5064|(7, 6)|var|float32
output = func_5063(var_5064)
func_5065 = relay.Function([var_5064], output)
mutated_mod['func_5065'] = func_5065
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4889_call = mod.get_global_var('func_4889')
func_4891_call = mutated_mod.get_global_var('func_4891')
call_5120 = func_4889_call()
call_5121 = func_4889_call()
func_3359_call = mod.get_global_var('func_3359')
func_3361_call = mutated_mod.get_global_var('func_3361')
const_5138 = relay.const([4.560550,1.119553,-0.211535,4.469455,-9.880992,-2.168656,4.903502,8.698647,0.710603,-4.210403,9.472381,3.930867,9.320214,-6.963613,-2.708691,0.039162,0.142062,1.326467,-3.212109,-8.706490,-0.720855,6.328611,1.651580,-6.412999,-5.538232,8.080853,-9.621171,5.950916,-0.488936,-4.077042,2.920392,-4.662205,-9.189586,-9.245049,-5.385625,7.988419,-7.292279,-6.194150,3.078733,-4.481122,5.451338,-8.400430,-4.608617,-4.337153,6.480703,4.782523,-5.761735,5.230076,5.013252,4.200197,3.067039,-6.513882,-1.155538,1.455663,8.280780,5.662840,-0.716178,-0.674331,5.395692,0.153350,5.630674,2.787163,-9.702285,-6.432178], dtype = "float64")#candidate|5138|(64,)|const|float64
call_5137 = relay.TupleGetItem(func_3359_call(relay.reshape(const_5138.astype('float64'), [2, 16, 2])), 0)
call_5139 = relay.TupleGetItem(func_3361_call(relay.reshape(const_5138.astype('float64'), [2, 16, 2])), 0)
output = relay.Tuple([call_5120,call_5137,const_5138,])
output2 = relay.Tuple([call_5121,call_5139,const_5138,])
func_5143 = relay.Function([], output)
mod['func_5143'] = func_5143
mod = relay.transform.InferType()(mod)
output = func_5143()
func_5144 = relay.Function([], output)
mutated_mod['func_5144'] = func_5144
mutated_mod = relay.transform.InferType()(mutated_mod)
var_5158 = relay.var("var_5158", dtype = "bool", shape = (15, 7, 9))#candidate|5158|(15, 7, 9)|var|bool
const_5159 = relay.const([[[False,False,False,False,True,False,True,True,True],[True,False,True,False,True,True,True,False,False],[False,True,False,True,False,True,True,True,True],[True,False,False,True,False,True,True,False,False],[True,True,True,False,False,False,False,False,True],[True,True,False,True,False,True,True,True,False],[False,False,True,False,True,True,False,False,True]],[[True,True,True,False,False,True,True,True,True],[True,True,False,True,False,True,True,True,True],[False,True,True,False,True,True,False,True,True],[True,False,True,True,True,True,False,True,True],[True,True,False,True,False,True,True,True,True],[False,False,True,False,True,False,True,True,True],[False,True,True,True,True,True,True,False,False]],[[False,False,False,True,False,True,False,True,True],[True,False,True,True,False,True,False,False,False],[False,True,False,True,False,True,False,True,False],[True,False,False,False,False,True,False,False,True],[True,False,False,False,True,False,True,False,False],[True,False,True,True,False,False,True,False,False],[True,False,False,True,False,False,False,False,True]],[[True,False,True,False,True,True,False,True,False],[False,True,True,False,False,True,True,True,False],[False,True,False,False,False,False,False,True,False],[True,True,False,False,False,False,True,False,True],[False,False,True,False,False,False,True,True,False],[True,False,True,True,False,False,True,True,False],[True,True,False,False,False,True,False,False,False]],[[True,True,False,False,True,True,True,False,True],[True,True,False,True,False,True,True,True,True],[True,False,False,False,False,False,False,False,False],[True,False,True,True,True,False,False,False,False],[True,True,False,False,True,True,True,False,False],[False,True,True,True,True,False,False,True,False],[False,True,False,False,False,False,True,False,False]],[[False,False,False,False,True,True,False,True,False],[True,False,True,True,False,False,True,False,True],[True,False,False,True,False,True,True,False,False],[True,False,False,True,False,False,True,False,True],[True,False,False,True,True,True,False,True,True],[True,False,True,False,True,True,True,False,True],[False,True,False,False,False,False,False,False,False]],[[False,True,False,True,False,False,False,False,True],[True,False,False,False,True,False,False,False,True],[True,True,True,True,True,False,True,False,True],[True,False,True,True,True,False,True,False,True],[True,False,True,False,True,True,True,False,True],[True,False,True,True,True,False,False,False,True],[True,True,True,False,False,False,True,True,True]],[[True,False,True,False,True,False,True,False,True],[True,True,True,False,False,True,True,True,True],[True,False,True,False,True,True,False,True,True],[False,False,True,True,True,True,False,False,True],[False,True,True,False,False,False,True,False,False],[True,True,False,False,False,False,True,True,True],[False,True,False,False,True,False,True,False,True]],[[False,False,True,True,True,True,False,True,True],[False,False,False,True,True,True,True,True,False],[True,True,True,False,True,False,False,True,True],[False,False,True,True,True,True,True,False,False],[False,True,True,True,True,False,True,False,True],[True,True,True,True,False,False,False,False,True],[False,False,True,True,False,True,False,False,False]],[[True,True,True,False,True,False,True,False,True],[True,False,False,True,True,True,False,False,False],[False,False,False,True,False,True,False,True,True],[False,True,True,False,False,False,True,False,True],[True,False,False,True,True,False,True,False,True],[False,False,True,False,False,True,True,False,True],[False,False,False,True,False,True,True,True,True]],[[True,False,False,True,False,False,True,True,True],[True,False,False,False,False,False,True,True,True],[False,False,True,True,False,True,True,True,False],[False,False,True,True,False,True,True,True,True],[True,False,False,True,True,False,True,True,True],[True,False,False,False,False,False,True,True,False],[True,False,True,True,False,False,False,True,False]],[[True,True,True,True,False,True,True,True,False],[False,False,True,True,False,True,True,False,True],[True,False,False,True,True,False,False,False,False],[False,False,True,False,True,False,True,False,True],[False,False,False,False,False,True,True,True,True],[False,False,True,True,True,True,True,False,False],[True,False,False,True,True,False,True,True,False]],[[True,True,False,False,True,False,True,False,False],[False,True,True,True,False,True,False,True,False],[False,True,False,False,False,True,True,False,False],[True,True,True,True,False,False,False,True,False],[False,False,False,True,False,True,False,True,True],[False,True,True,False,True,False,True,False,False],[False,True,False,False,False,True,True,False,False]],[[False,False,True,False,False,True,False,True,True],[True,False,False,True,False,True,False,False,True],[False,True,True,False,True,False,True,True,False],[False,True,False,False,True,False,True,True,True],[False,True,False,True,False,False,True,True,False],[True,True,True,False,True,False,True,False,False],[True,True,True,True,True,False,False,False,True]],[[False,True,False,True,False,True,False,True,False],[False,True,True,True,False,True,False,False,False],[False,True,False,False,False,True,False,True,True],[False,True,True,False,False,False,False,True,False],[False,True,False,False,False,True,True,True,True],[False,True,True,True,True,False,True,False,False],[True,False,True,False,False,True,True,False,True]]], dtype = "bool")#candidate|5159|(15, 7, 9)|const|bool
bop_5160 = relay.logical_and(var_5158.astype('bool'), relay.reshape(const_5159.astype('bool'), relay.shape_of(var_5158))) # shape=(15, 7, 9)
output = relay.Tuple([bop_5160,])
output2 = relay.Tuple([bop_5160,])
func_5166 = relay.Function([var_5158,], output)
mod['func_5166'] = func_5166
mod = relay.transform.InferType()(mod)
var_5167 = relay.var("var_5167", dtype = "bool", shape = (15, 7, 9))#candidate|5167|(15, 7, 9)|var|bool
output = func_5166(var_5167)
func_5168 = relay.Function([var_5167], output)
mutated_mod['func_5168'] = func_5168
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4770_call = mod.get_global_var('func_4770')
func_4772_call = mutated_mod.get_global_var('func_4772')
call_5172 = relay.TupleGetItem(func_4770_call(), 0)
call_5173 = relay.TupleGetItem(func_4772_call(), 0)
uop_5175 = relay.asin(call_5172.astype('float32')) # shape=(14, 2, 10)
uop_5177 = relay.asin(call_5173.astype('float32')) # shape=(14, 2, 10)
output = relay.Tuple([uop_5175,])
output2 = relay.Tuple([uop_5177,])
func_5178 = relay.Function([], output)
mod['func_5178'] = func_5178
mod = relay.transform.InferType()(mod)
output = func_5178()
func_5179 = relay.Function([], output)
mutated_mod['func_5179'] = func_5179
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4889_call = mod.get_global_var('func_4889')
func_4891_call = mutated_mod.get_global_var('func_4891')
call_5180 = func_4889_call()
call_5181 = func_4889_call()
output = call_5180
output2 = call_5181
func_5182 = relay.Function([], output)
mod['func_5182'] = func_5182
mod = relay.transform.InferType()(mod)
mutated_mod['func_5182'] = func_5182
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5182_call = mutated_mod.get_global_var('func_5182')
call_5183 = func_5182_call()
output = call_5183
func_5184 = relay.Function([], output)
mutated_mod['func_5184'] = func_5184
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5143_call = mod.get_global_var('func_5143')
func_5144_call = mutated_mod.get_global_var('func_5144')
call_5190 = relay.TupleGetItem(func_5143_call(), 2)
call_5191 = relay.TupleGetItem(func_5144_call(), 2)
output = relay.Tuple([call_5190,])
output2 = relay.Tuple([call_5191,])
func_5200 = relay.Function([], output)
mod['func_5200'] = func_5200
mod = relay.transform.InferType()(mod)
output = func_5200()
func_5201 = relay.Function([], output)
mutated_mod['func_5201'] = func_5201
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5200_call = mod.get_global_var('func_5200')
func_5201_call = mutated_mod.get_global_var('func_5201')
call_5284 = relay.TupleGetItem(func_5200_call(), 0)
call_5285 = relay.TupleGetItem(func_5201_call(), 0)
func_3359_call = mod.get_global_var('func_3359')
func_3361_call = mutated_mod.get_global_var('func_3361')
call_5299 = relay.TupleGetItem(func_3359_call(relay.reshape(call_5284.astype('float64'), [2, 16, 2])), 0)
call_5300 = relay.TupleGetItem(func_3361_call(relay.reshape(call_5284.astype('float64'), [2, 16, 2])), 0)
func_417_call = mod.get_global_var('func_417')
func_419_call = mutated_mod.get_global_var('func_419')
const_5318 = relay.const([-7.391034,-6.771964,-0.374329,-7.748356,6.479809,7.016817,4.087205,-3.213035,-8.035385,-6.165647,4.573258,-5.244888,-2.360265,-7.581052,-6.665216,8.215517,8.624851,1.874759,-5.204748,-4.292930,4.686700,-6.505287,-2.353931,1.166854,2.379143,2.418370,7.729319,-8.987126,-5.514485,-9.709393,-3.700881,2.623780,3.666054,1.625211,-7.984751,9.842533,-1.255251,-5.266454,5.744301,9.797646,-7.000765,6.083975,5.838161,5.342273,8.446852,-1.303232,0.700861,2.338409,-3.549517,-8.905182,-6.344655,-9.904774,0.040978,7.558108,-8.365563,6.176014,-2.564801,2.339417,8.485040,3.462580,-7.574314,-2.205517,2.474120,-4.469216,8.422078,5.877610,-7.218261,-2.763810,-8.717358,-7.313890,-3.691038,9.467778,3.983700,-2.280918,9.775630,9.393514,4.632244,-2.045991,6.263074,-6.833289,2.126401,-2.656726,-2.956610,-1.931741,-2.890212,1.408253,-0.763666,7.229410,3.634170,-0.198593,0.745518,-9.924811,-0.756154,-2.785457,0.671968,-1.114835,-2.541059,3.259007,9.784684,3.044551,-9.473113,-7.145075,-1.806380,-9.667074,-5.189155,-1.171219,-8.755219,-1.973908,1.115041,-7.215311,-9.978227,-0.223550,4.659306,9.052192,7.039235,3.976348,8.474458,-8.857414,1.997418,-3.883245,-7.872741,-0.891854,2.068193,0.064805,8.546782,-6.110587,-2.852198,-3.591380,-5.362181,-7.769847,-9.513591,7.965435,2.928308,7.671650,6.019209,-0.588895,6.513548,-3.337674,-4.492608,4.100878,-7.879506,8.176460,-1.855969,-0.776087,4.629807,-6.370533,-0.581103,9.094613,-3.387854,4.464735,-8.459605,9.263547,-9.113193,3.302558,9.407975,-4.227659,-4.084124,-6.905610,6.152246,1.178591,4.067330,-8.693370,6.468428,1.224237,9.381757,3.495459,7.148338,1.742528,8.876500,-5.400437,0.037924,-8.719932,-5.314832,4.600990,0.885169,-1.423052,-7.258860,9.262691,-7.329971,6.286352,6.300665,-3.860007,4.755906,-2.425763,-3.720305,5.563996,9.681496,4.311736,-7.727577,2.302227,8.349771,-2.310103,-2.608951,4.937896,5.971681,-3.165913,-8.955502,5.108060,-3.476722,-7.202971,-5.741190,1.040870,0.001566,0.367071,7.788407,5.015206,-6.635039,-2.601380,3.282550,8.231211,9.613861,0.102476,-7.853512,9.511337,-4.484370,3.189315,-1.064300,-2.222050,4.560910,2.489535,-4.690672,0.735985,0.151241,-5.866358,-5.259880,3.700766,9.468067,6.375985,9.323447,8.390261,-7.998171,-6.283074,9.598228,-6.340901,-2.286620,4.396554,-4.810487,6.848086,-2.639689,3.503235,4.075089,-7.682304,2.348487,5.569876,6.645114,3.641168,9.699285,-5.199523,-0.132271,-5.866684,-5.296980,-7.663794,-5.669115,-0.621094,0.880560,8.274288,-5.348570,0.598088,7.762764,-2.906992,3.498867,-6.310634,4.305808,6.932283,3.828338,-5.930447,6.880607,5.028784,-7.401958,8.836323,1.157094,5.265135,-5.585384,9.547986,7.653952,-0.811116,0.874997,0.333745,8.660725,8.542457,-7.268545,6.414983,-6.123870,-3.527507,-8.367635,-3.174476,-7.145219,-4.838509,-0.255199,9.393281,4.648868,9.870115,-5.243421,-8.432675,8.112827,7.767345,4.936761,-9.965108,0.436575,3.652514,-4.382441,5.177194,7.621821,8.972964,2.232957,7.863922,1.921835,-6.723092,2.454703,5.387341,-7.404940,4.788836,9.015084,9.368140,-2.488359,9.053158,-6.925839,0.228985,-4.218808,4.520698,7.495755,-7.004680,6.444880,1.612681,9.943823,-2.556808,9.888059,-9.844600,7.608836,0.951309,-2.320735,-0.425042,1.040218,-1.684831,-9.315032,6.553297,-7.146447,-2.711313,-8.717573,-2.177344,2.364805,5.689134,-7.155517,7.326323,4.841990,2.773610,3.181429,-7.460771,-9.575936,-7.778193,6.833581,-9.391369,-7.817618,6.424175,-1.623573,-3.100691,-6.566362,-6.004103,3.547775,-8.886607,-2.850574,-4.878929,3.409115,-8.316931,9.659389,4.346861,6.933027,-0.362019,-7.227819,-8.864750,-4.866572,-4.565920,5.611977,8.635593,0.799439,7.473856,-0.485135,8.496889,9.500273,9.062950,6.565661,9.363152,-0.370620,-0.982547,8.818555,-1.246933,9.501897,8.830487,3.518852,-5.050248,-1.662040,4.320883,8.138455,6.453525,-7.630722,-3.552807,6.535256,-6.641058,4.140186,-9.841199,-8.258158,-4.261355,-2.926631,-1.201738,8.067481,-9.224593,-0.146117,-7.687397,-5.773485,-8.224224,-8.263134,-1.456077,-4.934242,9.484617,6.007383,0.464601,-3.385373,-4.528369,6.450871,5.632399,5.931559,-3.593169,3.741605,8.895436,-7.631644,-8.988617,-3.207051,-9.513541,-4.313443,1.314842,-8.416465,0.082790,-8.201715,4.603186,-8.369113,-5.199795,-5.378003,8.790449,5.940472,8.979281,7.515481,-4.134970,6.417957,-9.920312,1.458692,9.442385,-7.770312,-9.001949,-4.147076,-7.892092,-0.319571,-3.117773,-5.828778,-7.900841,0.986595,3.014351,-0.835857,-2.059711,8.992418,6.819845,-4.712052,-6.347634,4.693510,-0.660271,-2.755920,-1.304363,3.252637,0.184212,-1.619845,-7.095690,7.485242,-7.971604,3.779822,4.919684,-4.073142,1.197386,-9.873395,8.396820,5.779038,7.961959,2.977041,-3.136974,5.130173,-9.805528,-6.642277,3.938397,-0.217463,9.565055,-6.365492,-6.935695,2.123722,-4.323765,7.844672,6.924970,2.795680,-7.317124,-9.458003,6.215686,-8.613992,7.298226,-5.711365,1.824254,7.642511,7.070573,1.889234,-1.271915,5.656631,-5.997472,-0.167656,6.390249,0.950158,-1.825111,5.187135,3.355520,0.573254,-9.535682,8.739191,5.024929,-2.058263,-1.876459,-9.177208,-9.443159,3.511611,3.387329,-1.370987,0.419341,-2.206631,6.312360,-6.105315,-8.636457,3.635225,9.947842,-5.644827,0.542995,7.531135,-4.538908,-1.208075,-8.059502,-1.185339,-1.498408,-1.848042,-8.714945,3.260647,-6.770571,-1.817676,-0.335898,2.171317,-5.879581,-5.687445,-2.732514,4.567720,7.126077,0.911323,-5.640646,0.938377,-5.221276,-0.989104,6.387407,-5.044546,1.151459,-7.874564,3.072688,-8.009300,5.152459,0.834899,-4.797278,-6.385026,1.320953,4.451668,8.452528,-0.862271,4.320664,-7.915236,-5.782019,-4.485522,5.935039,2.205736,5.197919,-5.429503,-2.392447,-0.430400,8.167288,7.568614,5.436269,-3.613811,4.384976,-3.197392,3.874975,6.995456,-3.871857,-0.612421,-0.897629,-4.437798,3.051117,-6.692858,-0.350667,-1.602677,-7.232927,-0.383862,8.524711,-7.779695,4.571273,4.829651,-0.170771,4.348742,-6.557841,-3.864927,3.763972,-9.207632,-1.655095,8.593037,-2.856020,-8.925207,0.262153,-2.182838,1.821423,1.515821,6.509260,6.497822,-9.899513,-3.359347,-0.229721,6.027315,4.999320,-5.656782,7.588141,8.651270,-5.415696,-7.260965,-4.897400,-9.115164,1.798547,0.572839,-1.018519,-9.231171,7.421361,8.633856,-3.762220,1.078005,4.921826,8.144975,1.920338,-8.613150,-8.948264,0.716729,5.154778,-0.589712,-2.092125,9.333791,-5.145599,4.161905,2.710529,-8.160900,8.848513,1.684151,3.734371,0.989347,9.366627,-6.735152,-0.261616,6.099836,2.178405,4.022559,2.099444,-8.759410,-3.927921,4.730197,0.428332,5.895580,1.582340,1.751998,4.706003,-6.536311,-4.577505,-3.369345,-7.903449,-5.230050,7.253960,7.188172,-0.545128,0.396648,3.429945,-9.713520,-9.671609,-3.463605,2.388174,-3.186080,-4.900277,8.888673,-2.662848,-2.899538,5.829147,-3.843372,-9.485865,3.006947,-2.707378,1.257146,1.063037,-0.927268,-5.276602,-3.496439,-9.654291,-9.162025,8.312207,6.007138,-3.784368,-6.224223,-5.635540,-0.785403,9.855059,7.911651,-0.460977,3.715356,7.793938,8.234661,-3.929294,6.071553,8.170706,-3.991551,4.224552,-1.276081,5.223573,-9.469078,4.715792,-6.203271,-5.336289,-1.299230,-4.496411,2.127034,4.078612,1.783175,-0.029319,-3.404759,1.715570,-7.367607,-0.079511,-4.452613,8.792535,4.940860,0.772454,0.630966,-6.790247,6.674652,-2.789499,-3.268340,0.322503,4.827365,-7.363945,5.721531,-5.728183,-3.307041,-8.493087,-3.591995,3.533616,3.133555,0.608797,3.362532,-6.866438,0.609529,2.711080,7.221227,8.722664,-0.266358,-3.845314,4.239982,5.180626,-0.027956,-1.267678,0.433955,-5.812423,7.339018,-2.521175,6.915916,5.168390,3.326074,-7.179353,5.391019,9.319319,6.383016,-2.302794,4.161091,1.202403,7.989557,-6.483054,5.179451,1.197760,7.799107,0.189796,1.736045,8.919784,0.571071,8.942893,3.057500,-2.118916,-8.026955,-0.554685,-5.188647,4.106600,4.826009,2.754727,3.129399,6.920753,-2.691263,0.833722,3.779730,-2.989037,4.191067,-4.309722,-1.318251,3.260795,0.369339,-7.681816,8.822559,6.267867,-5.415182,-6.040323,-0.394224,-6.325881,-2.796091,-9.445568,-0.453156,4.249736,-4.618625,-1.920506,-0.802051,4.225391,7.379389,-0.365711,-4.941123,-3.638143,-5.085153,-1.666765,-1.294701,7.604819,-6.128574,7.792366,-1.133154,-7.373060,-2.410775,4.730146,8.994319,-8.537660,8.059855,9.496403,-2.777669,-9.763532,-3.090434,9.643163,0.942052,-1.235484,-9.665279,-2.097199,-8.606194,3.091929,9.130426,0.293313,-7.510147,-5.484045,-2.845370,5.899937,0.444878,6.888750,-4.348711,-5.711486,6.380450,7.317346,-3.872648,-4.489155,4.871401,3.271531,3.360877,8.699356,-9.437989,1.290620,0.902022,-5.005127,-3.295999,2.295485,-4.677226,-8.501424,-0.642616,-2.920760,5.541345,4.820288,-5.995583,3.149514,0.339942,5.770317,-9.615940,5.427072,7.898745,-7.914763,5.437740,-2.182456,-4.293336,-5.702634,7.933899,5.218014,6.800398,6.788253,-9.747522,2.894859,2.606730,4.255634,7.819735,6.353701,-8.841264,-1.283943,8.363173,-4.106149,8.182499,-1.483146,1.682627,7.615143,-5.904986,-8.380757,-7.456431,9.465729,1.156846,3.093113,1.976184,3.320364,2.870763,7.254456,9.994143,-9.598006,-4.868166,2.990710,-7.232886,6.412592,1.923663,1.861625,7.186054,9.840446,-5.939131,-2.718670,5.126311,4.247564,6.193299,-8.733545,-4.851175,-4.793361,8.896508,-8.198682,-7.435170,8.781755,-6.537687,0.875123,2.243197,-8.263096,-4.685710,0.210489,-8.957690,-8.873033,7.435993,-2.150233,7.620448,8.460681,-6.307931,-0.968884,7.490620,-2.521664,6.997951,3.583750,-8.223600,-5.799209,-5.889581,0.965231,-4.889521,1.795111,-8.099899,-7.366748,-4.816848,-5.177648,5.174106,2.980026,-2.823192,-4.266012,-0.603963,-0.521967,2.554848,2.617579,-1.268258,8.463134,-5.004225,2.075770,-2.903605,-9.276086,9.309515,-8.878215,2.885035,1.530083,3.963447,6.797500,4.315730,2.969223,-8.215460,5.033211,8.700676,-1.683465,3.080842,-3.432326,5.830702,-7.299602,3.127922,1.118849,-4.999710,1.882343,5.579691,6.115605,-4.421490,9.565729,3.092591,-2.330440,9.938438,-9.339986,-6.151464,6.600393,8.211690,-6.621659,-8.996716,-0.223729,-5.408575,-8.693414,-0.794107,-3.695536,-3.483768,-3.585967,-2.474887,-2.165150,8.569454,-1.390661,2.151613,4.336863,2.339047,-3.041495,5.488112,6.003804,-1.798993,4.096762,9.214001,7.557047,-6.626086,-8.916704,-8.078031,-2.154101,4.641776,9.253492,-9.925917,-7.293944,-0.760165,1.038560,2.917495,0.814332,8.969517,-9.929096,0.734632,-2.593691,9.169872,-3.646782,-1.797865,6.663675,-5.199579,-3.066053,1.416085,-5.314879,4.117037,-7.494468,-2.927180,-4.449270,-8.525621,3.258294,8.390555,3.278170,-6.635497,-0.649379,6.688363,-7.850825,0.716451,-3.486031,8.341664,-8.286310,2.876699,1.742629,7.880054,4.671512,8.023520,7.644396,-6.687038,6.175500,-4.867286,7.334456,-4.820785,2.582086,0.275886,-8.714404,5.513347,8.386687,0.455214,-5.343447,4.031706,5.965050,-0.076493,0.578990,1.655026,8.308747,-7.293803,5.900230,8.611100,-7.142825,2.721960,4.704279,-4.426670,2.665474,-0.855426,-2.763156,8.802570,-1.971623,-5.004514,-3.644206,-5.758237,2.924114,4.336380,7.185178,-9.388682,5.156792,1.441863,-5.178646,5.865414,4.904746,4.923692,9.445769,9.079731,4.545059,5.804143,-2.051154,6.204651,3.419752,-5.789930,4.069280,-7.959883,-8.017416,-0.409925,5.572581,-4.216824,7.815877,-2.254528,-1.023994,-9.878714,3.263084,-7.911898,6.100170,-6.444819,-4.162385,-4.061665,4.079254,-9.278243,-3.231430,-2.219142,-1.867845,-0.154968,-9.126122,3.389308,3.147185,-6.967388,8.083380,-7.877345,4.735032,0.850321,9.807188,9.431599,2.637822,2.852955,-0.442088,6.185742,-3.974041,-8.007045,0.609269,-8.913985,1.017643,5.214414,1.811963,-8.503071,-3.659057,-7.746147,1.060789,-7.376353,7.488824,-2.043379,2.447697,-1.728101,5.908584,7.053943,-7.014991,0.592863,-4.469032,3.028510,-0.295014,-3.935444,4.073702,0.974389,6.330228,0.160006,2.939777,-4.548269,-4.629028,-7.039856,-9.080905,-0.778725,2.461969,-0.816761,-1.524369,4.969684,0.971465,-4.304569,-9.875691,0.966448,5.490278,-1.093040,-8.836761,-7.582988,-6.210297,7.574773,-9.731578,7.041177,9.869104,2.582773,6.402883,1.448013,5.270056,-0.255201,4.736206,2.860071,-0.910240,8.345449,-8.512228,8.865766,-1.339982,8.458267,-5.930527,1.559495,1.249485,-2.776612,4.268685,1.347951,0.187619,5.807968,-3.910743,7.088768,-2.918687,8.320876,-4.611650,3.003209,9.544403,7.167780,-3.680318,-0.541820,-5.210367,-9.419596,-3.063312,-9.633633,5.492256,-5.757191,-8.431653,3.471021,-7.000848,3.377147,0.054763,1.913471,-8.395906,8.398052,3.655414,-8.614868,-2.905927,-4.906682,-3.187966,-2.727616,-5.309174,8.376901,-3.936912], dtype = "float64")#candidate|5318|(1280,)|const|float64
call_5317 = func_417_call(relay.reshape(const_5318.astype('float64'), [8, 10, 16]))
call_5319 = func_417_call(relay.reshape(const_5318.astype('float64'), [8, 10, 16]))
output = relay.Tuple([call_5284,call_5299,call_5317,const_5318,])
output2 = relay.Tuple([call_5285,call_5300,call_5319,const_5318,])
func_5332 = relay.Function([], output)
mod['func_5332'] = func_5332
mod = relay.transform.InferType()(mod)
output = func_5332()
func_5333 = relay.Function([], output)
mutated_mod['func_5333'] = func_5333
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5143_call = mod.get_global_var('func_5143')
func_5144_call = mutated_mod.get_global_var('func_5144')
call_5339 = relay.TupleGetItem(func_5143_call(), 0)
call_5340 = relay.TupleGetItem(func_5144_call(), 0)
func_3139_call = mod.get_global_var('func_3139')
func_3143_call = mutated_mod.get_global_var('func_3143')
var_5342 = relay.var("var_5342", dtype = "float64", shape = (75,))#candidate|5342|(75,)|var|float64
var_5343 = relay.var("var_5343", dtype = "float64", shape = (525,))#candidate|5343|(525,)|var|float64
call_5341 = relay.TupleGetItem(func_3139_call(relay.reshape(var_5342.astype('float64'), [5, 15, 1]), relay.reshape(var_5343.astype('float64'), [5, 15, 7]), ), 0)
call_5344 = relay.TupleGetItem(func_3143_call(relay.reshape(var_5342.astype('float64'), [5, 15, 1]), relay.reshape(var_5343.astype('float64'), [5, 15, 7]), ), 0)
func_3359_call = mod.get_global_var('func_3359')
func_3361_call = mutated_mod.get_global_var('func_3361')
var_5346 = relay.var("var_5346", dtype = "float64", shape = (64,))#candidate|5346|(64,)|var|float64
call_5345 = relay.TupleGetItem(func_3359_call(relay.reshape(var_5346.astype('float64'), [2, 16, 2])), 0)
call_5347 = relay.TupleGetItem(func_3361_call(relay.reshape(var_5346.astype('float64'), [2, 16, 2])), 0)
func_5166_call = mod.get_global_var('func_5166')
func_5168_call = mutated_mod.get_global_var('func_5168')
const_5350 = relay.const([False,False,True,False,True,True,True,True,True,True,True,True,False,False,False,False,False,False,True,False,False,True,True,True,True,False,False,False,False,False,False,False,True,True,False,False,False,False,True,False,True,True,True,True,True,True,True,True,False,False,True,False,True,False,True,False,True,True,False,True,False,False,False,True,True,False,True,False,False,True,False,True,False,True,False,True,False,False,False,False,False,False,True,False,False,False,False,True,False,True,True,False,True,True,True,True,True,False,True,False,True,True,True,True,False,False,False,False,False,True,True,False,True,False,False,True,False,True,True,False,False,False,False,True,True,True,False,False,False,True,False,True,False,False,True,False,False,True,True,False,False,False,False,True,False,True,True,False,False,False,True,False,False,True,True,True,True,False,True,True,True,False,False,False,False,True,False,False,True,True,False,True,True,False,True,True,True,False,False,True,False,True,True,False,False,True,True,True,True,False,False,False,False,True,False,False,False,True,False,True,False,False,True,False,True,False,True,False,False,True,False,False,False,True,False,True,False,True,False,True,True,True,False,False,False,False,False,False,True,True,False,False,True,True,False,False,True,True,True,True,True,True,True,True,False,True,False,True,False,True,False,True,False,False,True,False,True,False,False,False,True,False,False,False,True,False,True,False,False,False,True,True,True,False,False,True,True,True,False,True,False,True,True,False,True,False,False,False,False,False,True,True,True,True,False,False,False,True,True,False,True,False,True,False,True,True,True,False,False,False,False,False,True,True,False,False,True,False,True,False,True,False,True,False,True,True,False,True,False,True,True,True,False,False,True,True,False,True,True,False,True,True,True,False,False,True,False,False,False,True,False,True,True,True,True,True,True,True,False,True,True,False,True,True,False,False,False,False,True,True,True,False,True,False,False,True,True,True,True,True,False,True,False,False,True,True,True,False,True,True,True,False,True,False,True,False,True,True,False,False,False,True,True,True,True,True,True,True,False,False,False,True,False,True,True,True,False,False,True,True,True,False,True,True,True,False,True,False,True,True,False,False,True,True,True,False,True,False,True,True,True,False,False,True,True,True,False,True,True,True,True,True,True,False,False,False,True,True,False,False,False,True,False,True,False,False,False,True,False,True,True,True,True,True,False,False,True,False,False,False,True,True,True,True,True,True,True,False,False,False,True,True,True,True,False,True,True,False,True,False,False,False,True,True,True,False,False,False,False,False,True,False,True,False,True,False,False,False,True,False,False,False,True,True,True,False,True,True,False,False,True,False,False,False,False,True,False,False,False,True,False,True,True,True,True,False,True,True,False,False,False,True,False,True,False,False,True,True,True,False,True,False,False,True,False,False,True,True,False,True,False,False,False,True,True,True,True,True,True,True,True,True,False,False,False,True,False,False,False,True,False,False,True,False,True,True,True,False,False,True,True,False,False,True,True,True,True,False,False,False,False,False,True,False,False,True,True,False,True,False,True,True,False,False,False,True,True,True,False,False,False,True,False,False,True,False,False,False,False,False,False,False,False,True,True,False,False,False,False,False,False,True,True,False,True,True,False,True,False,False,True,True,True,True,True,False,True,True,False,False,False,False,False,False,True,True,True,False,False,True,False,False,True,True,True,False,True,True,True,True,True,False,False,False,False,False,True,True,True,True,True,True,False,False,False,True,True,True,True,True,False,True,False,True,True,True,False,False,False,True,True,True,False,False,True,False,False,False,True,True,False,True,True,False,True,True,True,False,False,True,False,False,False,False,False,True,True,False,True,True,True,True,False,True,True,True,True,True,False,True,False,False,False,True,False,True,False,True,True,False,False,True,True,True,True,True,False,False,True,False,True,False,True,True,False,False,True,True,True,True,False,True,True,False,True,True,True,True,False,True,False,True,False,True,False,False,False,False,False,True,False,False,True,True,True,True,True,False,True,False,True,True,False,False,False,True,True,True,False,False,False,False,True,True,True,True,True,True,True,True,False,False,True,False,True,True,True,True,True,False,True,True,False,True,True,False,False,True,True,True,True,False,True,False,True,True,True,False,False,False,True,True,False,True,True,False,False,False,True,True,True,False,False,True,False,False,True,False,True,False,True,False,False,True,False,True,False,True,True,False,True,True,True,False,False,False,False,False,True,True,True,False,True,False,True,True,False,False,False,True,False,True,True,True,False,True,False,False,False,False,True,True,True,False,True,True,True,True,True,False,False,False,False,False,False], dtype = "bool")#candidate|5350|(945,)|const|bool
call_5349 = relay.TupleGetItem(func_5166_call(relay.reshape(const_5350.astype('bool'), [15, 7, 9])), 0)
call_5351 = relay.TupleGetItem(func_5168_call(relay.reshape(const_5350.astype('bool'), [15, 7, 9])), 0)
func_5000_call = mod.get_global_var('func_5000')
func_5005_call = mutated_mod.get_global_var('func_5005')
var_5369 = relay.var("var_5369", dtype = "float32", shape = (792,))#candidate|5369|(792,)|var|float32
var_5370 = relay.var("var_5370", dtype = "uint16", shape = (8, 42))#candidate|5370|(8, 42)|var|uint16
var_5371 = relay.var("var_5371", dtype = "int8", shape = (729,))#candidate|5371|(729,)|var|int8
call_5368 = relay.TupleGetItem(func_5000_call(relay.reshape(var_5369.astype('float32'), [792,]), relay.reshape(var_5370.astype('uint16'), [336,]), relay.reshape(var_5371.astype('int8'), [729,]), ), 0)
call_5372 = relay.TupleGetItem(func_5005_call(relay.reshape(var_5369.astype('float32'), [792,]), relay.reshape(var_5370.astype('uint16'), [336,]), relay.reshape(var_5371.astype('int8'), [729,]), ), 0)
output = relay.Tuple([call_5339,call_5341,var_5342,var_5343,call_5345,var_5346,call_5349,const_5350,call_5368,var_5369,var_5370,var_5371,])
output2 = relay.Tuple([call_5340,call_5344,var_5342,var_5343,call_5347,var_5346,call_5351,const_5350,call_5372,var_5369,var_5370,var_5371,])
func_5374 = relay.Function([var_5342,var_5343,var_5346,var_5369,var_5370,var_5371,], output)
mod['func_5374'] = func_5374
mod = relay.transform.InferType()(mod)
mutated_mod['func_5374'] = func_5374
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5374_call = mutated_mod.get_global_var('func_5374')
var_5376 = relay.var("var_5376", dtype = "float64", shape = (75,))#candidate|5376|(75,)|var|float64
var_5377 = relay.var("var_5377", dtype = "float64", shape = (525,))#candidate|5377|(525,)|var|float64
var_5378 = relay.var("var_5378", dtype = "float64", shape = (64,))#candidate|5378|(64,)|var|float64
var_5379 = relay.var("var_5379", dtype = "float32", shape = (792,))#candidate|5379|(792,)|var|float32
var_5380 = relay.var("var_5380", dtype = "uint16", shape = (8, 42))#candidate|5380|(8, 42)|var|uint16
var_5381 = relay.var("var_5381", dtype = "int8", shape = (729,))#candidate|5381|(729,)|var|int8
call_5375 = func_5374_call(var_5376,var_5377,var_5378,var_5379,var_5380,var_5381,)
output = call_5375
func_5382 = relay.Function([var_5376,var_5377,var_5378,var_5379,var_5380,var_5381,], output)
mutated_mod['func_5382'] = func_5382
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4857_call = mod.get_global_var('func_4857')
func_4858_call = mutated_mod.get_global_var('func_4858')
call_5402 = relay.TupleGetItem(func_4857_call(), 0)
call_5403 = relay.TupleGetItem(func_4858_call(), 0)
uop_5409 = relay.log(call_5402.astype('float32')) # shape=(14, 2, 10)
uop_5411 = relay.log(call_5403.astype('float32')) # shape=(14, 2, 10)
output = relay.Tuple([uop_5409,])
output2 = relay.Tuple([uop_5411,])
func_5416 = relay.Function([], output)
mod['func_5416'] = func_5416
mod = relay.transform.InferType()(mod)
mutated_mod['func_5416'] = func_5416
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5416_call = mutated_mod.get_global_var('func_5416')
call_5417 = func_5416_call()
output = call_5417
func_5418 = relay.Function([], output)
mutated_mod['func_5418'] = func_5418
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5332_call = mod.get_global_var('func_5332')
func_5333_call = mutated_mod.get_global_var('func_5333')
call_5434 = relay.TupleGetItem(func_5332_call(), 1)
call_5435 = relay.TupleGetItem(func_5333_call(), 1)
output = call_5434
output2 = call_5435
func_5442 = relay.Function([], output)
mod['func_5442'] = func_5442
mod = relay.transform.InferType()(mod)
mutated_mod['func_5442'] = func_5442
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5442_call = mutated_mod.get_global_var('func_5442')
call_5443 = func_5442_call()
output = call_5443
func_5444 = relay.Function([], output)
mutated_mod['func_5444'] = func_5444
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4857_call = mod.get_global_var('func_4857')
func_4858_call = mutated_mod.get_global_var('func_4858')
call_5480 = relay.TupleGetItem(func_4857_call(), 2)
call_5481 = relay.TupleGetItem(func_4858_call(), 2)
var_5487 = relay.var("var_5487", dtype = "float64", shape = (660,))#candidate|5487|(660,)|var|float64
bop_5488 = relay.less_equal(call_5480.astype('bool'), relay.reshape(var_5487.astype('bool'), relay.shape_of(call_5480))) # shape=(660,)
bop_5491 = relay.less_equal(call_5481.astype('bool'), relay.reshape(var_5487.astype('bool'), relay.shape_of(call_5481))) # shape=(660,)
output = bop_5488
output2 = bop_5491
func_5513 = relay.Function([var_5487,], output)
mod['func_5513'] = func_5513
mod = relay.transform.InferType()(mod)
mutated_mod['func_5513'] = func_5513
mutated_mod = relay.transform.InferType()(mutated_mod)
var_5514 = relay.var("var_5514", dtype = "float64", shape = (660,))#candidate|5514|(660,)|var|float64
func_5513_call = mutated_mod.get_global_var('func_5513')
call_5515 = func_5513_call(var_5514)
output = call_5515
func_5516 = relay.Function([var_5514], output)
mutated_mod['func_5516'] = func_5516
mutated_mod = relay.transform.InferType()(mutated_mod)
var_5536 = relay.var("var_5536", dtype = "int8", shape = (8, 6, 11))#candidate|5536|(8, 6, 11)|var|int8
const_5537 = relay.const([[[-5,2,-4,-6,-7,1,-3,-2,-3,-7,-7],[7,-8,4,-10,9,-5,5,5,-6,-2,-8],[5,-2,-10,-3,9,-6,-10,6,-3,7,-5],[-1,-4,3,-1,-4,2,5,6,-3,-7,-1],[-9,8,3,-4,-10,1,10,-1,-6,9,1],[-3,3,4,-5,8,-2,-10,4,5,8,9]],[[9,-1,2,-4,6,-2,3,1,-8,4,5],[-7,9,-4,-6,-7,2,-2,9,-9,-3,-4],[-6,-6,3,-4,-7,9,3,-3,8,-5,-9],[5,3,3,-5,-4,-1,5,4,4,2,-7],[1,-7,-1,9,1,8,-10,-2,9,5,-8],[-5,4,4,-5,3,1,4,4,2,7,-9]],[[-4,-3,-9,-4,-1,-9,-10,-10,-2,-10,-5],[-4,8,-4,9,-4,-4,-5,-5,10,-10,10],[-2,3,-5,8,9,-7,-10,-2,4,-4,-2],[3,-10,5,4,7,8,-1,-7,3,10,1],[-2,-9,-10,7,3,-1,-5,-2,4,6,-1],[4,-5,-4,-8,-5,9,-7,8,-2,5,2]],[[2,2,3,-7,-4,-4,10,4,-4,-10,-2],[5,2,10,3,-1,8,-10,-1,7,9,-6],[1,9,-8,10,6,1,3,5,-5,-10,10],[-4,-8,-6,-4,6,-4,-5,-1,-4,2,-3],[1,2,6,9,-3,-7,-7,6,7,9,4],[-6,8,-2,-7,5,6,-4,-1,-10,-9,5]],[[-2,-7,4,2,9,5,-8,-8,-3,-5,-2],[3,5,8,-10,9,-6,3,-2,8,-5,-9],[-1,9,1,2,4,-7,-9,-2,9,-1,6],[5,-2,-9,7,9,-6,1,-4,-3,8,-7],[8,-6,-9,-2,-4,4,1,3,3,7,4],[-9,2,-7,5,-7,-3,9,-1,-8,-9,3]],[[6,-9,9,8,-5,2,-10,-8,-2,7,-10],[-9,7,-10,-10,8,6,-4,-5,-8,9,8],[8,-4,2,8,-8,-7,-6,1,7,-3,7],[9,2,-2,9,2,-7,10,8,3,8,-2],[6,-4,3,7,-7,-3,8,-9,7,-6,8],[6,-5,5,-9,-10,10,-10,-1,4,5,-8]],[[3,10,1,-4,2,-8,-9,2,7,4,6],[-2,-3,9,-2,-1,-6,-4,-8,7,-6,-9],[-5,-4,-5,-4,4,-9,2,5,5,-8,-8],[8,-1,-8,-10,-4,7,1,-8,-2,1,-9],[9,7,-3,-7,-7,9,-1,-7,-3,8,-10],[-2,4,-1,5,-7,8,-2,-9,2,-3,-6]],[[-7,5,-3,-9,-2,1,5,10,-6,8,-9],[4,7,-5,-8,-7,3,6,5,-1,6,-2],[6,5,-8,-5,-1,-4,5,-1,2,10,9],[3,1,2,8,10,-5,7,6,-10,-10,-7],[6,10,5,-5,-7,-8,-4,6,4,9,1],[-3,-7,5,-7,-4,10,1,5,-10,4,-1]]], dtype = "int8")#candidate|5537|(8, 6, 11)|const|int8
bop_5538 = relay.maximum(var_5536.astype('int8'), relay.reshape(const_5537.astype('int8'), relay.shape_of(var_5536))) # shape=(8, 6, 11)
func_5000_call = mod.get_global_var('func_5000')
func_5005_call = mutated_mod.get_global_var('func_5005')
var_5544 = relay.var("var_5544", dtype = "float32", shape = (792,))#candidate|5544|(792,)|var|float32
var_5545 = relay.var("var_5545", dtype = "uint16", shape = (336,))#candidate|5545|(336,)|var|uint16
var_5546 = relay.var("var_5546", dtype = "int8", shape = (729,))#candidate|5546|(729,)|var|int8
call_5543 = relay.TupleGetItem(func_5000_call(relay.reshape(var_5544.astype('float32'), [792,]), relay.reshape(var_5545.astype('uint16'), [336,]), relay.reshape(var_5546.astype('int8'), [729,]), ), 0)
call_5547 = relay.TupleGetItem(func_5005_call(relay.reshape(var_5544.astype('float32'), [792,]), relay.reshape(var_5545.astype('uint16'), [336,]), relay.reshape(var_5546.astype('int8'), [729,]), ), 0)
output = relay.Tuple([bop_5538,call_5543,var_5544,var_5545,var_5546,])
output2 = relay.Tuple([bop_5538,call_5547,var_5544,var_5545,var_5546,])
func_5572 = relay.Function([var_5536,var_5544,var_5545,var_5546,], output)
mod['func_5572'] = func_5572
mod = relay.transform.InferType()(mod)
var_5573 = relay.var("var_5573", dtype = "int8", shape = (8, 6, 11))#candidate|5573|(8, 6, 11)|var|int8
var_5574 = relay.var("var_5574", dtype = "float32", shape = (792,))#candidate|5574|(792,)|var|float32
var_5575 = relay.var("var_5575", dtype = "uint16", shape = (336,))#candidate|5575|(336,)|var|uint16
var_5576 = relay.var("var_5576", dtype = "int8", shape = (729,))#candidate|5576|(729,)|var|int8
output = func_5572(var_5573,var_5574,var_5575,var_5576,)
func_5577 = relay.Function([var_5573,var_5574,var_5575,var_5576,], output)
mutated_mod['func_5577'] = func_5577
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5332_call = mod.get_global_var('func_5332')
func_5333_call = mutated_mod.get_global_var('func_5333')
call_5596 = relay.TupleGetItem(func_5332_call(), 0)
call_5597 = relay.TupleGetItem(func_5333_call(), 0)
func_417_call = mod.get_global_var('func_417')
func_419_call = mutated_mod.get_global_var('func_419')
const_5639 = relay.const([[-0.042055,7.072928,-6.750468,2.620869,1.865008,-2.799045,-8.804296,-6.064624,8.109148,-9.528208,4.473193,0.493965,-5.980855,-1.123731,3.673416,-0.577371,-4.586399,-0.904855,7.219010,-3.101263,4.260469,-1.336907,-5.372852,-6.666410,-6.018283,9.870931,-3.139211,9.201930,6.314348,-0.073625,-8.149227,-7.108010,-5.923928,6.523239,-5.665859,-8.785068,4.403022,5.345409,4.941478,0.302177],[5.954612,2.349918,0.106931,-0.887737,-5.267398,9.640741,3.156499,-9.016812,-1.300678,1.070329,9.912943,-5.260144,-5.865965,-3.672817,3.397066,-2.795621,-7.765459,3.703675,-7.105242,-9.602601,-6.031149,7.765040,8.587741,8.640794,2.465927,-2.601415,-8.288027,1.021062,-1.023496,-2.117592,3.974373,-1.482388,8.577653,0.828003,4.983023,-1.693201,-2.823268,2.624930,5.710586,1.909689],[-3.041251,0.214613,2.904335,5.583495,-0.229776,3.872574,-7.286472,0.915349,1.894752,0.431628,-7.316139,-8.770455,3.020413,-2.515759,-4.671388,-8.188842,2.675389,7.059325,-7.506440,0.931526,0.070031,6.889855,-6.418229,1.878526,3.766560,2.147960,-1.600928,-2.259967,5.896195,1.497141,2.158027,-5.203202,8.262894,0.404543,4.068109,4.922924,2.723476,-0.736038,3.543420,-5.309883],[4.274647,3.111955,2.490051,6.566047,-0.293877,9.893017,-0.595342,7.580397,7.924968,-0.194272,8.195650,8.193419,-6.469617,-3.595216,0.838233,-8.638017,-4.992567,8.521101,-6.070413,3.081552,8.561258,-9.592958,-1.688788,6.416394,-1.286605,-9.057646,7.010756,-8.615197,-2.479749,-3.446987,4.949267,-1.595315,-5.715779,-4.242443,-2.401107,3.128298,-5.845538,-2.860825,5.622987,-9.216643],[8.704267,4.775909,0.889861,-8.245496,4.880336,3.210963,8.634037,-4.659862,8.999192,-2.289098,-5.220688,1.723160,6.679515,-9.620909,-0.156318,-8.297945,4.850671,-9.773582,-7.020474,-7.336158,8.414170,-0.195141,8.939075,-7.709841,6.832126,0.895704,9.499002,-2.998015,-0.870302,7.460044,-3.137731,-9.180569,-5.665956,-6.462159,0.438269,-3.902702,-1.745878,8.656767,5.789961,6.862215],[2.397941,0.502282,5.135764,7.173833,-6.425171,-9.323841,6.083480,-9.580760,-4.245881,-9.097047,-6.270715,-8.858070,8.856186,-3.820812,-7.094698,-2.077960,-9.804343,-4.101784,7.649527,4.351229,-7.005648,-0.370127,-7.660246,5.978304,5.378729,-6.368528,-8.546986,5.089336,7.027097,4.838881,8.198931,6.273678,8.303297,3.420609,1.964291,-5.166529,1.931131,7.238145,0.200057,-0.655180],[-6.659787,-3.727313,2.032907,3.056793,-0.680378,-4.077115,-0.569028,3.097655,2.634286,-6.714085,-4.871457,0.051304,-3.690438,5.444057,7.665096,-3.281520,2.667192,8.099212,-6.842246,0.337735,5.413508,-9.363347,5.514846,6.173405,4.731441,7.625643,-2.424453,6.713436,-1.568890,-2.424549,-2.386813,6.549366,-5.243650,-5.349072,4.899307,9.153856,-7.389686,-8.236978,7.001152,-6.853429],[-4.853497,-8.837836,-6.693072,6.719854,6.136122,5.635336,-9.568424,8.212482,5.629441,-9.890202,4.941512,9.346020,6.727202,5.360477,9.637281,3.262387,6.761779,-9.302470,9.086102,-0.522239,8.084907,-2.153257,8.118652,-3.640449,-1.411548,0.044961,-1.340330,2.809904,3.855968,0.661151,-3.909126,-3.304637,-0.385518,-6.258468,-9.893738,-7.324527,-6.924346,0.543947,-3.791821,-1.156476],[-2.104619,-7.632772,-2.453199,0.784117,-2.334094,-6.485693,6.379481,8.580094,6.375118,-9.400127,-3.513504,9.391181,5.288117,-7.552335,2.419133,-2.974375,8.369346,-0.365743,-9.585181,2.310760,-5.392784,-7.090010,-8.697785,-8.882532,8.800824,9.433203,4.074622,-1.356129,3.282153,3.286351,0.324026,2.830004,-4.909408,-2.764033,7.391498,-6.631653,2.844558,5.874709,-7.008743,0.293192],[-7.232768,1.856220,8.598238,3.388559,5.159437,0.830822,-8.672344,5.575281,3.706538,-6.700990,4.638515,-0.081149,3.870986,-6.221143,2.100696,2.164073,-2.692770,-1.931492,-9.857521,-8.959337,-0.234349,-3.167404,-1.597510,0.962765,9.723582,-6.245730,-4.417667,2.013129,3.394457,4.564458,2.584651,-6.510564,-3.862027,2.073731,-1.057595,-1.802444,-8.615973,-4.873310,-3.208017,-5.906414],[4.140115,-9.898022,9.645514,3.226729,1.771564,-8.147588,5.599952,-8.930969,7.093041,-1.327661,7.512497,-7.184560,3.806175,2.346124,-1.333945,9.893534,-3.880584,-3.344457,1.642122,-4.917454,-8.408131,7.037876,-7.510872,5.715755,5.740706,0.569202,-7.633785,-1.316098,-2.187741,3.736793,4.615823,-8.655204,-1.331893,-2.192772,-9.648873,-4.658010,1.655811,-5.087262,4.737961,3.138184],[3.036626,-3.905446,-8.931522,7.874128,-0.984173,1.169068,4.684995,9.450601,-4.140895,9.521478,2.324733,-2.717668,1.506954,-8.180241,-2.720381,-0.309487,-9.539657,-8.920161,3.864391,-4.722417,-8.197388,-0.252214,1.195473,-9.123872,2.924832,9.650590,8.974112,2.156498,-0.990476,-6.049904,4.084968,5.526612,6.346747,-0.998069,1.994232,-9.647234,-2.464996,-1.009235,-7.462925,-0.566640],[-0.524409,-0.918133,8.273676,7.672716,4.731973,-1.170274,-1.329834,1.309641,0.798028,-9.545438,-4.161012,2.958644,3.382702,-0.681416,-4.859807,6.205719,-3.328957,-3.723799,0.030943,-4.190008,6.289745,0.944793,-2.723529,0.322070,-2.884221,-9.120731,-0.814304,-0.037912,6.277198,-5.584493,9.632085,4.947000,9.046153,1.589094,8.826257,-9.610064,-3.068846,-0.275932,-3.184700,2.789782],[6.658816,1.143250,2.431066,-6.629363,-7.845766,-0.653444,-6.489870,-5.400566,7.035822,4.566996,6.741063,-8.667033,8.967331,-2.137339,-2.024466,0.688381,8.666916,-2.134949,-0.956324,-2.755954,6.431635,-6.448414,-8.749811,4.765432,-3.870049,-8.648499,-9.798138,5.374277,7.408502,6.085761,9.472403,7.543785,0.413911,0.113085,-8.666380,9.810536,-8.914364,-8.058633,-2.189602,0.734608],[-0.744252,-7.501136,-0.975851,-6.599966,7.624909,3.932564,-0.624789,3.445146,-6.116695,7.191186,-2.642445,-0.346278,-8.471087,3.249131,8.655358,8.590084,-9.706216,-9.503826,-8.726322,-0.109981,0.504600,-5.036619,1.085967,2.867013,5.317284,-8.491578,3.715000,-2.678284,5.383192,-4.979616,-5.974508,-3.235844,7.483939,7.475227,-7.725679,9.447624,-6.733207,2.137173,-9.965088,3.041577],[-0.175952,1.345643,9.415184,7.829812,-7.396021,7.864046,-5.759429,7.909151,-4.565476,0.715032,8.674313,-9.485102,-6.221097,4.062339,6.833973,-8.937430,-0.956825,4.249888,-1.961696,0.929510,7.694859,7.817567,-5.597604,9.504852,-6.876736,0.002691,-1.848369,4.565475,-7.123195,3.752384,3.990620,9.893706,7.483953,2.403460,5.394528,-6.928653,8.843420,-3.720351,6.464854,7.756524],[-8.323698,2.244148,-8.751474,-1.271596,8.110917,-8.835225,9.053591,-0.562775,7.024187,-3.231247,-0.631173,9.217440,1.593756,-2.444237,6.300916,-8.100284,-8.653078,-5.963849,-7.533896,9.615178,-6.958314,-9.364985,2.706997,2.753816,0.864889,-3.130361,-8.075934,4.357651,3.656528,-1.039137,1.941854,3.378934,-4.102311,3.248040,-5.290936,-3.681323,-8.561855,-0.932250,-0.085473,-9.021213],[-0.459608,3.579096,-0.120258,-5.715067,-6.939267,7.256995,9.031455,-3.744244,-9.276607,-6.160064,-7.478091,-5.162324,9.407336,4.147453,-6.981606,9.035083,-3.567101,1.599053,-4.274302,-3.854888,8.433483,-9.851522,-6.373401,-4.715512,-1.331806,3.148282,-4.329392,-3.454413,2.725945,7.863835,9.326199,-5.866033,8.934002,3.481605,6.141996,6.385520,6.318835,5.429897,-6.179748,-6.317742],[-9.931942,2.196071,4.931251,-0.700803,0.794553,-5.374076,8.037224,-3.202066,6.577665,-8.479638,-6.771589,3.367288,1.453929,1.276002,-8.035205,2.518383,-3.742808,8.765627,-8.617232,-4.100359,1.037986,-3.036893,9.840422,-0.753510,-3.888800,5.171400,-7.242565,-4.375471,2.279086,0.802859,-7.210145,-9.381591,9.123307,-7.125113,3.170489,-6.300707,-7.805117,9.259807,0.827256,-9.473384],[-8.937758,-0.522145,-9.972631,0.609794,9.838745,1.201929,0.010506,-0.978254,-8.122827,-3.200668,1.477629,-6.877295,5.747369,4.843477,6.093036,5.639247,3.612479,-8.868180,-1.206878,8.913288,-5.873134,1.726282,-2.741826,1.441737,5.538611,-0.444166,-0.843883,-8.019893,0.632929,-1.746759,4.223729,-9.050580,-9.214489,-9.311015,-1.191655,-9.500513,-7.480235,-5.582281,7.995892,4.706150],[6.707774,2.327735,8.674433,-1.400582,5.675779,0.458275,-8.432547,2.097008,7.328120,9.277931,5.489899,-7.168450,-0.202056,-9.739268,-5.006951,-1.013551,6.226309,4.295823,4.677166,8.767277,8.946425,8.880309,8.449230,-1.696459,-3.964316,-5.630325,7.661668,8.397452,-4.326196,9.207357,-6.611036,-6.815804,0.017670,-4.298405,5.505969,0.577407,-1.167134,-2.571181,5.297834,-3.466240],[1.777332,3.901223,-9.449958,-1.767437,-5.758624,6.414689,5.255112,-1.568962,-1.338696,9.671721,4.038088,-1.409513,9.588276,1.303306,1.341384,-5.971051,-9.156030,1.965073,-5.108964,4.198093,3.946436,8.245109,4.951272,0.753075,-3.592445,7.348734,7.411720,-3.023164,8.088515,4.607640,6.143006,6.748673,3.455753,-7.789905,-4.126348,-9.431556,1.963011,-9.915914,1.224823,-5.528735],[-7.103942,0.850361,9.795099,-8.624765,3.975434,1.957794,9.080003,3.517925,5.085046,-4.137366,4.514604,7.126409,2.039616,2.400577,2.229480,-5.028363,-8.891841,6.068948,5.978406,-5.292165,9.624432,-3.557823,-9.889804,4.481834,3.126416,0.792139,-6.415485,2.984188,-5.969610,-5.760637,-6.767082,8.698622,5.974334,-2.109458,-3.088529,3.909646,-9.046623,-1.059516,3.432060,-5.859040],[2.573735,-8.072823,-2.900100,7.322215,-9.390801,-8.603422,9.227100,8.273706,-0.948976,6.035927,-9.240271,-6.202984,-2.671230,0.111289,-3.772832,0.220114,1.458157,8.216014,1.102925,-0.993296,1.881846,8.736179,-7.441391,7.756907,-0.325740,4.655672,0.542382,-9.185110,-9.094448,-6.461104,1.106136,0.718463,2.691870,6.894386,6.989955,-7.061264,-2.774792,0.194619,6.780119,-6.138357],[-2.229087,-3.643837,4.790301,0.038104,9.187022,2.194138,5.970556,1.342204,3.594875,-8.479581,8.398491,9.557166,-1.305052,-4.074285,6.030782,7.687251,1.873415,-7.629926,-7.338318,-5.852953,8.487873,0.968989,-9.544320,2.086028,1.900269,2.193176,-3.112451,-0.998981,7.698137,-9.631012,9.457289,-6.022804,-1.429112,-3.669629,8.302906,-4.268665,-8.643127,3.340053,-8.559822,-3.095022],[-3.088764,-7.425777,1.316310,5.259257,-6.929141,0.602380,-7.974914,-2.274315,-0.756065,7.254442,2.464278,-5.640528,9.598720,6.698059,-6.212245,5.560275,8.679719,-0.314315,8.125664,8.435748,2.467480,5.337081,9.336482,-3.307043,-9.418811,9.566837,8.620853,7.554355,9.394353,6.584726,0.245635,1.716497,-7.882844,7.444105,3.849418,2.316657,7.331703,-9.614703,-5.348523,1.401094],[9.587188,4.459354,-3.631349,6.822140,7.310913,-6.653476,-2.777463,3.374018,4.805601,-7.069231,6.294833,4.252510,-9.223974,3.495288,5.807040,-4.288151,8.679629,1.246549,-6.712225,-6.706234,3.199202,-3.424631,-1.731359,5.433645,-0.814960,-4.737974,5.471648,6.273266,-4.690828,8.347220,0.533617,-3.914918,2.435181,-5.431756,-7.443514,3.987780,2.733510,6.388325,2.758263,0.336561],[-1.606943,7.181862,-7.419717,-1.497197,-5.339866,-7.760773,9.723928,-6.513893,-5.020173,-3.177883,3.213217,5.381429,-4.711140,-4.496574,0.038057,9.086568,-9.523109,6.045663,6.121388,9.898017,7.385708,1.124321,-1.485655,8.827983,7.700509,4.947392,5.494339,6.630969,-5.674916,7.058310,-4.397989,3.290697,-6.394580,0.707121,-0.624008,7.632060,-9.314311,-9.551707,-1.931352,0.107618],[-9.243043,9.793835,9.269663,4.246454,-3.541461,1.194377,8.590510,2.367550,-6.994557,-9.227935,7.965026,-5.720027,-1.258678,-0.457672,7.539451,-0.285633,7.136201,2.248432,2.240567,-4.720312,-4.675004,-3.353501,-9.029548,3.179546,-7.307099,-7.209863,8.881985,5.783097,3.397476,-0.804631,9.514942,-0.862675,-3.362311,-2.739416,-6.723728,-2.515424,-3.875413,8.309316,-1.919272,8.320296],[-1.394877,5.795524,4.019535,2.705070,-8.151504,1.350315,9.479582,-7.381436,4.834571,3.409006,8.524361,-0.988653,-0.416397,0.407005,-6.189382,5.695489,-5.076977,-4.918372,9.526631,-1.789666,-8.303411,1.577749,-9.741364,8.172560,-1.565227,0.983474,-4.216948,-5.052054,2.977508,-4.693578,-1.070938,-3.155554,-5.007835,-7.789003,2.236365,4.678181,-0.880631,8.638535,5.946489,8.195288],[9.249833,8.064673,-9.067861,-0.358605,6.883143,-2.166902,-1.639883,-3.265090,7.405905,-2.008403,-2.653018,-2.455517,-4.236893,4.833728,1.330403,-5.107418,6.978033,-6.800658,6.746614,-0.576549,6.105820,-8.684225,5.704889,-1.393717,-8.932693,5.519755,0.989106,5.737766,9.569556,0.450314,6.127610,8.924033,0.701058,8.562576,-4.761021,-7.158989,-9.344579,4.879594,5.111679,-7.111765],[-3.734532,-2.497226,-3.479378,-7.909155,-1.042661,-9.629812,7.308965,5.425832,8.798348,8.014730,-3.202662,-1.672378,-6.575425,-3.236907,0.810767,2.099393,9.388794,3.544355,-0.164771,3.438993,9.165862,-5.608488,-0.744983,-2.143132,-3.035774,2.231317,9.388679,7.578285,-8.011809,3.776839,5.666561,2.968751,7.334075,-9.404864,-9.839714,-5.221719,9.344878,-7.402369,-9.497267,6.599508]], dtype = "float64")#candidate|5639|(32, 40)|const|float64
call_5638 = func_417_call(relay.reshape(const_5639.astype('float64'), [8, 10, 16]))
call_5640 = func_417_call(relay.reshape(const_5639.astype('float64'), [8, 10, 16]))
output = relay.Tuple([call_5596,call_5638,const_5639,])
output2 = relay.Tuple([call_5597,call_5640,const_5639,])
func_5642 = relay.Function([], output)
mod['func_5642'] = func_5642
mod = relay.transform.InferType()(mod)
mutated_mod['func_5642'] = func_5642
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5642_call = mutated_mod.get_global_var('func_5642')
call_5643 = func_5642_call()
output = call_5643
func_5644 = relay.Function([], output)
mutated_mod['func_5644'] = func_5644
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4770_call = mod.get_global_var('func_4770')
func_4772_call = mutated_mod.get_global_var('func_4772')
call_5698 = relay.TupleGetItem(func_4770_call(), 0)
call_5699 = relay.TupleGetItem(func_4772_call(), 0)
func_1497_call = mod.get_global_var('func_1497')
func_1499_call = mutated_mod.get_global_var('func_1499')
const_5715 = relay.const([1.883134,-5.687012,-9.533589,0.923861,4.666404,-9.177126,5.664387,-2.376411,0.682934,0.671304,2.189907,3.440024,-6.897677,-6.629170,-3.579462,-6.618374,-0.108573,1.316544,2.880950,9.103686,-2.243713,-9.261867,-6.398322,-2.176211,-4.841423,1.714866,5.846501,-9.655946,-3.928726,-8.983308,-1.669427,6.832025,1.850621,-7.194362,3.290175,8.150618,2.520967,-2.528040,-3.464637,-6.178663,-4.116632,-6.050670,-3.747545,1.974825,2.622153,-0.627719,-1.083615,-2.214648,-8.571026,-1.732017,9.749936,-7.344794,6.645383,-7.234184,6.628861,-0.675820,0.430785,-4.599478,2.479485,-4.345875,4.270062,3.281296,-5.482329,1.797253,-3.447377,8.348536,-5.996628,-3.798485,-0.614756,7.578472,9.841112,0.026183,1.606738,9.362122,9.949175,2.727165,-6.753469,-7.397457,1.676570,9.916926,7.592699,7.477833,-7.331818,-3.420301,-7.927047,5.916402,8.062682,-7.561604,-3.743989,-0.871510,-0.581678,-8.918398,-2.834983,3.382069,-7.653599,4.171132,9.948436,7.060266,-5.661443,-9.898479,-8.797435,7.786079,-1.501699,-7.278288,5.213577,-3.562258,-4.306644,-0.264279,6.215587,-4.444874,-5.625230,-6.125510,-9.372062,-6.997130,8.539602,-6.744125,-2.118040,1.302534,0.559251,4.822564,-6.452435,-3.097724,-5.890772,0.120156,1.086657,7.113368,6.430863,5.236459,-6.207417,8.598020,0.193738,9.608353,1.253008,-1.439152,-6.502164,7.813817,-8.099898,6.952596,-5.919753,-3.273746,5.031725,0.043543,9.979880,-0.782590,-7.281367,6.838132,9.353130,5.217171,-8.290191,-2.935681,7.931731,8.116392,5.651969,-1.421968,-8.009082,6.220947,1.690699,-7.921065,-4.585749,-5.946581,-4.689165,-2.529747,-2.896482,-5.833249,4.222566,0.628899,-6.916754,-4.456025,9.675408,1.978099,-8.919274,1.061043,-9.428613,-3.029638,0.245936,5.897130,6.505338,1.422034,-2.406369,-7.711790,-8.191082,-9.805544,-5.836936,4.959425,-9.717957,3.735555,6.884095,-1.788648,9.596712,-6.367843,9.542830,-8.391511,0.659951,-7.795715,-4.476344,-3.020456,-9.776453,7.698904,2.893203,-5.630001,-3.076812,-2.190120,5.359124,0.863562,-5.597457,6.367635,2.014818,-2.409457,0.025154,-8.805549,0.177807,8.570413,-3.588184,-4.903326,4.168912,8.088638,0.479718,0.971520,-3.650201,9.870830,2.816983,-3.340174,-6.977933,2.638974,9.283007,3.520879,-1.240650,-5.053196,-2.255915,5.120721,5.792968,3.631905,-3.212082,3.168752,9.989934,6.520422,6.331407,9.646189,-9.986011,-9.167701,2.817599,-8.347239,-1.279397,-6.246322,-8.798645,-9.176581,-3.236714,-4.110376,-2.685532,0.473955,-1.325449,-6.388447,-4.545381,5.352244,0.390314,-4.852249,-4.202686,-5.388769,0.264763,9.084031,-0.158569,9.501528,4.561628,-6.018771,-9.966993,5.763601,-7.733641,-8.340117,4.745253,-4.981915,4.630023,6.447300,3.665390,-5.797890,-4.367631,-7.230881,4.574613,5.558056,0.110601,4.955311,7.785174,-7.684534,-7.090322,-3.157491,4.212091,-2.283103,1.348644,-3.257395,-9.818997,0.910478,-6.427769,9.331425,-4.083956,-4.572762,-7.440185,8.668454,-6.496106,-6.889163,-8.059224,-6.304503,9.109584,8.035729,7.908613,6.623692,-5.556203,-7.788060,7.790939,9.030973,2.955355,9.876318,2.431105,2.621816,1.494925,-6.917683,9.554357,1.335753,-1.705171,4.770037,-9.960004,7.568594,-6.206191,1.165732,2.289171,0.845235,-6.786367,6.849635,3.510214,5.037145,-8.947595,-1.770608,-6.888594,-1.080870,-3.700650,-8.940768,-8.855882,-6.055151,-5.588862,-9.820405,-2.259411,-9.324831,0.107241,-5.280032,-8.577422,0.891199,6.559437,-3.436247,7.947604,-6.834099,1.142466,-1.570346,-7.596093,-7.945443,-9.515873,1.572453,-4.955202,2.160010,9.992231,-9.523515,8.528876,-5.234875,6.956576,-8.028808,8.803671,3.902092,-7.679213,-0.909835,-2.990165,7.763495,-7.453123,8.833187,-1.252983,-9.245446,-9.003239,9.913695,-5.179601,-5.966225,3.945433,-3.605047,-8.290174,8.138554,7.152235,3.700862,-4.713228,-8.978415,3.474329,9.153269,-4.646253,8.810331,-9.290771,-8.055993,-0.934230,-4.552813,6.182720,-7.822810,-6.332031,-9.233878,-5.271548,3.145357,-9.692316,9.278837,4.804483,5.611272,-0.015593,-6.551197,-0.865154,9.192860,-6.022657,-1.643778,-3.153332,4.187573,-0.220252,-1.106540,6.457433,8.175351,8.277004,-3.544364,3.333080,-5.966965,-8.188081,-5.991328,5.904212,-4.469540,7.013360,0.574384,3.416599,4.984766,1.204869,-4.034893,0.868767,-9.306383,-7.060044,-0.535379,1.478345,7.453708,4.812420,-3.633076,5.073452,-9.249303,1.553961,-5.865903,-2.466703,-2.009429,5.043821,5.785257,-5.204841,7.035771,-3.226484,-5.752039,-6.773583,0.939553,9.839229,-9.567842,-8.114885,2.650854,9.831536,7.357082,7.021694,2.553403,-6.171216,-4.234876,-8.487670,-1.910999,5.038480,-7.328564,-3.629037,2.233219,5.128537,1.820687,9.112206,3.283553,-0.053302,-8.019824,-7.952404,8.478958,-7.776064,9.864567,3.052351,8.841930,-5.539724,9.987297,-4.790774,-6.846326,-1.339302,1.931827,3.892780,-9.419952,7.351196,5.886008,-6.306077,0.031245,-2.684472,-3.731373,-8.455245,-6.559596,2.102427,0.712796,3.153858,-8.334257,2.684213,9.320546,-1.537246,5.955677,-9.975879,-2.985587,7.738078,-2.575952,-6.163484,5.292151,-2.477221,-1.227587,6.475715,3.438743,7.629459,2.356822,-3.621158,2.578072,-5.213063,-3.055814,-5.317772,-3.920638], dtype = "float64")#candidate|5715|(520,)|const|float64
call_5714 = relay.TupleGetItem(func_1497_call(relay.reshape(const_5715.astype('float64'), [8, 5, 13])), 4)
call_5716 = relay.TupleGetItem(func_1499_call(relay.reshape(const_5715.astype('float64'), [8, 5, 13])), 4)
func_5200_call = mod.get_global_var('func_5200')
func_5201_call = mutated_mod.get_global_var('func_5201')
call_5725 = relay.TupleGetItem(func_5200_call(), 0)
call_5726 = relay.TupleGetItem(func_5201_call(), 0)
func_4455_call = mod.get_global_var('func_4455')
func_4458_call = mutated_mod.get_global_var('func_4458')
var_5728 = relay.var("var_5728", dtype = "float64", shape = (441,))#candidate|5728|(441,)|var|float64
var_5729 = relay.var("var_5729", dtype = "int8", shape = (729,))#candidate|5729|(729,)|var|int8
call_5727 = relay.TupleGetItem(func_4455_call(relay.reshape(var_5728.astype('float64'), [7, 7, 9]), relay.reshape(var_5729.astype('int8'), [729,]), ), 2)
call_5730 = relay.TupleGetItem(func_4458_call(relay.reshape(var_5728.astype('float64'), [7, 7, 9]), relay.reshape(var_5729.astype('int8'), [729,]), ), 2)
var_5749 = relay.var("var_5749", dtype = "int8", shape = (729,))#candidate|5749|(729,)|var|int8
bop_5750 = relay.greater_equal(var_5729.astype('bool'), relay.reshape(var_5749.astype('bool'), relay.shape_of(var_5729))) # shape=(729,)
func_2016_call = mod.get_global_var('func_2016')
func_2019_call = mutated_mod.get_global_var('func_2019')
const_5763 = relay.const([-2.310570,6.079836,-6.796729,-2.511025,-7.339394,-9.231502,-5.073178,2.381223,-0.136386,4.953273,-9.714941,-2.480828,-0.438870,6.262708,-8.950909,-5.181694,-7.560565,8.632453,-9.022621,4.848497,-4.757767,8.366739,-0.438011,1.322480,-1.290184,-0.387132,-1.162779,-9.751367,-1.425585,3.791946,-4.042548,-9.317486,6.311589,9.759991,3.579483,-6.192921,7.433072,2.378825,-0.851782,-4.301309,8.152214,-5.670613,8.663889,-7.357274,-8.624500,9.691181,6.464428,1.816691,6.772208,2.750421,7.875659,-9.447438,0.931814,-0.965445,0.626494,-4.017512,8.739069,-7.700198,0.261714,9.754690,-5.153812,-3.086977,-5.264068,8.102221,9.607080,-4.884512,-9.004941,2.014177,-2.444977,-7.785513,-9.298275,-8.199815,0.810722,-4.943051,1.954581,-9.573166,-8.840361,-5.206973,-0.042206,-9.280321,-4.375408,8.006214,-5.179735,-7.381722,-4.523460,-2.542964,5.898524,9.337250,2.638843,2.343870,9.205249,3.790066,-7.686108,-2.835367,6.793617,-1.093301,5.614431,-2.884772,-7.713102,5.501117,1.565278,8.019933,-9.869974,9.354218,5.550312,5.825146,-9.213876,-6.363678,-8.466580,-6.257008,-3.768331,6.425324,-6.256011,2.339968,-0.025748,6.381023,-6.010356,-6.964644,2.176603,-8.831742,-2.843501,0.943545,-7.269487,1.578592,-8.832262,-5.173747,3.790851,3.824475,-2.995634,-3.295589,8.790031,-4.192144,0.963717,-0.431937,0.233117,2.674380,-4.053862,-0.027306,-8.750688,3.684600,4.758143,0.458332,0.236937,-3.692370,0.904861,-5.731636,1.461143,-5.408653,2.523007,5.847908,-2.482609,-1.708008,-0.064851,0.847989,3.598881,4.869992,1.624489,7.772533,4.452537,-2.772737,-3.230731,0.806357,9.523873,-7.670552,4.512223,1.679007,-9.953238,-5.323733,9.381950,-2.201884,6.078311,5.168504,-9.351026,9.651627,7.943282,0.179216,5.890627,1.927333,-7.561312,0.601652,2.546926,5.480468,7.173565,-9.610405,-2.355061,8.873749,6.614810,-4.950536,1.641373,7.203550,3.558756,-7.499198,-3.779302,4.013634,-4.191207,-2.285095,-6.585944,2.991621,1.041696,-9.945850,-0.990304,0.405014,-1.265759,-6.978540,-8.573784,-0.712017,-7.764282,-1.125188,-4.344267,6.925173,-7.835173,5.196970,-4.850818,-6.137203,-2.919018,1.488377,-6.370210,-1.444372,-1.838409,-4.571052,3.911721,-1.067109,-6.047206,-7.233831,-8.889333,5.635119,2.332211,2.944599,1.298337,0.598990,-4.444985,0.844125,-4.033071,-4.807274,-5.975317,1.052368,4.459284,9.477685,-3.437585,0.946804,1.022103,5.545537,6.417325,-1.652044,3.042532,-0.068158,-4.802571,-7.427389,-8.802953,0.890415,-8.381899,-6.960005,-6.015609,7.962872,8.516686,-8.557090,4.425970,1.658843,-0.688810,3.899076,5.254187,-9.283856,1.452601,-8.316213,2.019666,-0.260826,-0.121767,6.125206,-0.977529,0.794138,9.360751,9.568176,-6.553513,3.898583,-5.309951,-1.739759,6.632404,-7.137233,1.306634,-8.501897,-1.737669,-4.815032,7.582646,-4.061270,-4.583974,5.073285,6.699673,-4.711751,9.487909,-6.665724,7.645409,7.517253,3.886783,1.536369,0.741041,6.059515,3.845725,-5.063570,-5.862941,-2.189303,2.662288,-7.318592,-8.093082,-0.692475,5.529418,-9.313442,9.657552,1.872123,5.282485,2.450572,-8.056985,-6.749369,6.506653,-0.648379,6.449791,-9.999851,-1.872589,0.568184,-8.366507,-5.311439,6.371976,-9.863059,-2.041385,6.097390,-6.062477,-1.698510,-6.954718,-2.145740,-3.790962,9.538559,-6.822281,-4.446120,-6.815933,4.997266,3.157255,-7.245831,-8.641726,9.087156,-6.532443,-2.413647,4.546366,9.711219,-2.024045,2.637442,-7.023662,-9.187832,3.240254,4.342515,1.208921,-7.882454,-7.001260,2.723895,5.511808,-4.581367,8.887488,-4.326914,6.875255,-9.611320,-6.174883,-2.456184,-1.138140,4.720202,4.997401,-9.030105,3.455125,8.409214,8.286534,9.005355,0.291398,5.879895,-0.305569,-9.762954,2.195919,5.981889,-1.481425,1.277127,4.601083,-5.874261,7.940832,9.195992,8.336946,7.133278,3.952763,1.075926,-5.331731,3.903435,2.727772,-2.609557,5.260137,-7.308977,-3.813026,-4.718515,-4.265065,-6.302420,0.569711,-6.780249,9.987403,-6.068572,-4.768265,5.140187,-9.531308,-1.638216,-0.862001,-4.101184,-8.526686,-3.659261,-4.114810,-3.682614,5.108942,7.167322,-7.471464,-4.868426,9.104001,-4.552016,-4.096261,-6.356037,8.571435,-5.247525,8.278163,0.201134,0.795045,-9.626541,-2.881131,-3.183241,-6.735297,7.611619,-2.121336,-1.071661,-7.364594,5.608911,-7.176939,8.459693,-9.363395,-1.870050,-3.133287,-3.724106,4.163363,-2.247414,5.097482,9.892633,-9.329251,4.255659,-5.644957,-0.970970,-4.829560,-9.145622,8.108680,-0.785415,0.310086,2.621416,-6.744845,-0.221668,4.998032,-1.899449,7.162874,-7.061763,-4.048506,-3.069551,5.486628,7.188734,6.919265,-8.821011,8.255091,-6.479721,5.906699,-7.785414,2.174151,2.499344,2.497168,-3.310475,-6.677302,-6.648523,-9.825769,3.019206,1.643331,0.989794,-5.330713,3.709941,9.576274,0.184314,-1.251140,-3.936817,-5.030574,-0.734415,9.209922,-9.918899,1.086836,-5.281769,-1.524637,6.235924,-2.215203,-0.128390,-7.333770,-5.647246,3.966987,1.300868,-2.700128,-6.222540,1.992155,-3.963528,3.095849,-8.606330,-6.970478,-4.450545,9.576047,-0.176329,-1.192155,-2.859271,1.929979,-5.825201,-2.891983,-1.912643,-6.068382,0.225839,9.477649,-0.791454,5.057783,3.796253,-9.869555,-5.926758,0.924259,2.970337,9.366606,-4.948096,3.767825,7.603973,-3.488485,-5.812872,-8.277137,5.611284,2.638958,-0.073137,-4.748152,-4.576457,0.568803,-8.131834,-3.916565,4.707780,-5.041135,-2.275656,-5.030392,1.446021,1.314887,9.997627,-7.917454,2.827866,9.465945,-8.000795,8.039031,-0.044535,6.135174,4.140289,-8.368077,4.468754,-5.489943,-1.247650,3.009513,-2.485805,3.099977,3.964294,-2.569266,-9.187948,-9.788304,2.479287,6.542829,9.471602,9.036789,-0.701284,6.009457,-7.161326,2.765364,-9.860593,-9.101414,7.442878,5.294281,1.567511,1.238921,6.087155,-1.088494,4.938359,-5.453274,-0.375289,3.134063,0.498436,-1.680115,8.669137,7.325273,9.296701,-2.502455,6.800600,1.102766,7.585304,-3.717106,-5.018609,2.308716,2.381898,-2.152550,8.047730,-5.752627,-6.582280,-0.732695,5.910377,-6.113759,-2.028940,1.928471,-9.508599,-0.071013,-9.651784,-9.789368,-2.299033,-9.138993,0.964878,0.939954,5.051119,3.551306,0.829961,3.847810,1.486613,9.065337,-9.115008,-1.251495,4.308449,-1.300569,5.374046,-2.993789,-0.112787,6.260929,-9.811105,2.801672,9.778184,4.760690,-0.879237,5.113803,-8.220567,8.379153,9.147373,4.177434,8.187979,-2.363249,-0.981388,-2.858206,-7.142382,-1.688202,-0.730481,-3.715028,9.556714,-3.958355,-9.476792,1.059446,-2.319177,6.890184,-5.529211,-6.637339,-5.586820,9.474444,-0.298267,5.254026,7.336925,3.630499,1.174374,9.717272,9.716773,0.182334,9.528166,1.693352,7.296283,-7.086501,8.049591,9.310128,8.028367,2.819283,-8.695714,-9.213725,-4.954516,-4.108701,6.064492,-8.512481,3.579387,-4.348541,8.622122,3.120605,-2.207609,-8.812601,-8.577162,9.906924,-3.706070,0.741668,-5.062039,2.962390,1.051657,-1.305497,0.335816,-2.417067,-2.077779,-4.212662,-7.175035,1.066290,2.640553,-1.662557,1.835103,9.796589,-4.880504,2.428006,7.658127,-5.245278,-3.575424,9.052698,6.046659,-0.901025,1.116200,8.596748,-0.451061,-8.107296,1.756020,-5.623367,7.455012,-9.416693,-1.988336,-5.623310,-9.976122,1.847812,0.555210,6.860608,-0.107714,5.395180,-5.935543,0.790989,6.833441,-6.432966,-0.950561,7.541836,-4.129939,-8.207239,6.615976,-3.541143,0.213162,-0.620048,3.787481,3.779386,-2.352397,6.508433,-4.121114,4.845531,-9.727535,-9.774203,-6.156940,-3.190373,7.698981,-1.468794,9.554007,4.550609,-3.625365,-4.645743,2.490229,2.975495,-9.303311,-5.608882,0.122105,-6.194638,3.826386,-5.112404,-8.287693,-9.820599,1.749073,3.900544,-8.355331,9.958973,-1.416277,-1.809609,-4.508133,0.035323,-2.384206,8.654210,-5.084620,3.603369,-3.528257,0.040829,9.420944,-5.480588,-6.324913,7.460616,-2.887742,2.047350,-3.474858,-5.421866,8.918938,4.944753,9.596001,5.197328,-6.018982,0.890109,3.748396,-3.970655,6.125996,4.716446,-9.344042,-1.906929,-9.800382,2.331737,1.744319,6.868172,-9.617236,-6.608656,3.320051,0.266836,-3.387037,-8.900384,1.782175,-6.648714,8.281598,9.618172,5.391499,9.423022,7.415178,-1.380413,-9.673226,-2.091882,3.808061,2.607248,-7.532428,-1.831767,0.365393,-6.383543,9.141035,4.091497,7.797347,-9.758964,-6.734475,-1.672052,4.159444,-5.898108,-5.445937,-0.248273,9.578227,-7.757398,-4.203458,5.976675,9.383606,-4.219932,-9.750347,1.419034,-0.175939,-8.851567,6.414535,2.273290,-9.965789,8.761082,-7.984773,-9.258079,7.803489,9.263851,5.395979,3.786567,6.128214,-3.196983,-4.504771,5.168945,5.978620,9.901327,-1.013586,3.924042,3.817464,8.766164,-6.473534,4.961254,-2.783171,7.224507,4.126804,-4.047347,6.184301,1.322997,-4.792407,-5.227697,-4.888020,8.596661,-4.133325,6.625389,9.000780,7.107460,-6.318932,1.169570,-2.967357,0.497665,0.403589,-5.801371,-5.524556,-2.034772,-5.746223,-8.820136,3.689331,5.426504,6.060049,0.384137,8.356243,-2.613864,7.840003,-5.505717,6.161535,-2.077253,2.969681,0.661818,-5.773380,-9.157862,-9.567383,5.075482,-9.452832,-0.463557,-4.231290,-0.671665,-2.154629,2.766711,-5.561459,9.635666,2.664700,7.547317,-0.835108,-4.767714,-5.961306,9.254567,-7.813351,-6.120434,5.947887,-5.341459,-0.582119,8.525395,9.801118,8.820377,-2.119484,8.187192,-4.471943,-9.575586,7.198909,6.346867,-5.882359,8.149681,2.762013,2.115235,-2.251305,-9.033102,7.201076,0.032943,0.242708,-2.391868,-3.330979,3.425391,-3.991930,5.007032,4.242381,-9.740900,-4.722676,-8.876902,3.915429,-1.676623,5.954943,2.507289,4.084632,0.290249,3.926904,-0.704488,-7.863101,-1.065073,-3.868857,-8.865225,-8.157055,6.187183,-1.979175,-4.881562,0.563798,2.536650,-4.434765,8.774775,-2.815901,-6.111026,4.219933,2.249706,-8.669067,-1.767973,5.593815,-2.659899,8.602774,-2.991349,0.577196,-2.234089,-0.887876,6.136458,-3.783160,-1.229640,-6.569568,-9.426746,7.771954,7.883986,2.783366,-0.334176,-3.818543,-7.949411,-9.435265,-0.546008,6.185144,4.705100,-2.585519,-7.256711,2.800028,1.440172,3.693225,3.783495,6.737798,6.205870,1.810512,-0.745324,-8.027773,7.420224,9.847061,8.151953,-2.415941,-0.624880,8.442121,1.647161,3.107616,-5.833517,-0.287369,-3.569349,-0.141276,-6.615249,8.242081,-2.562924,7.565652,2.014705,6.504387,-3.982229,4.114531,-3.417478,7.109951,5.826810,2.524826,-4.015889,-2.528879,3.056650,-7.102821,-7.316790,-7.080437,8.658559,-9.685935,0.312162,3.447233,-7.288296,-9.918450,-3.839435,-1.378006,2.049761,0.732116,-1.480189,-2.984559,1.308186,4.651193,-5.013907,-3.935218,-2.867784,-7.238927,-6.277611,0.071928,-2.886190,6.589165,3.230031,5.622001,0.207439,-8.625026,-2.792414,-0.373271,8.019306,0.953838,7.468102,-3.057915,-3.634706,3.692305,-3.859522,3.682704,0.767835,8.158124,5.564300,3.527975,-9.689243,-0.158080,-0.475185,-0.183435,-9.450759,4.822286,5.110314,7.849428,5.103472,-5.770194,8.869146,-1.374588,-8.347916,-3.018723,-1.068615,0.205672,1.615199,4.410937,-7.514829,-5.001778,1.249221,-9.833620,3.148897,3.996729,0.453295,-4.583571,0.045041,8.682896,4.277417,4.279113,3.864433,-6.246798,-5.214185,-4.864360,6.507702,-4.352503,-1.103746,3.668018,-1.824745,-6.971431,-8.820769,7.131868,-9.439810,9.210476,-8.182157,6.409475,4.422948,-1.462164,-9.848330,-0.376247,6.456428,-0.643520,-0.764392,1.071556,-5.562518,-3.565931,1.077889,8.467366,-7.895294,4.460969,2.789219,-5.074333,-0.097893,1.166349,1.413787,-0.543045,-8.025038,-8.413532,-7.290004,9.606185,-2.846911,5.606988,9.555335,1.928477,5.773108,-2.885183,1.035197,-3.732204,-7.374823,-5.412633,8.441802,-0.011626,-7.529195,-2.427841,8.964075,7.950746,6.292134,2.508118,1.013539,-6.978876,5.938117,-9.533394,1.031516,-7.658696,-9.861838,4.799516,0.404744,3.065626,2.501145,-6.558538,2.280327,-0.431662,-3.992964,-5.225844,-7.007072,-4.036799,1.421641,-6.582568,4.124794,-3.206630,-9.476163,-6.383774,-1.490941,5.923392,0.379572,-9.790727,8.005599,-1.396112,-4.525331,-2.188397,6.395240,-7.783130,7.703869,-2.127726,6.249685,-6.911566,-7.980815,3.345702,-3.079954,-2.186215,-7.980852,-3.195921,4.161519,1.937185,6.070391,-7.726790,-3.568513,-8.711724,-9.405994,4.026652,6.380086,7.006402,-4.816993,-2.640184,5.808876,-1.682440,3.237108,8.525835,2.747705,-6.707540,-8.859890,-4.083043,1.045843,-1.478289,8.040256,8.233614,7.506392,-9.322621,1.681807,-5.134152,-7.224634,-2.460681,-6.674926,-5.054735,-0.694164,-1.291241,-9.286451,-9.108727,7.587887,1.269453,-3.977352,-6.578991,6.082721,1.768231,7.532268,6.123392,-9.101226,-5.352404,9.278687,8.580632,7.518613,0.662943,-0.817334,-5.310312,-2.744887,-8.752567,-2.576816,2.507126,7.856105,5.740969,-6.114799,-6.109384,-5.551201,2.988594,-1.753363,-0.965018,3.335123,6.581948,4.966752,0.292847,8.291861,-1.245601,-1.921020,9.428595,-7.866927,1.594618,8.215155,1.849332,-8.001213,8.626759,1.863099,-3.909211,9.327406,-3.359923,9.971012,3.669890,1.889247,3.671441,6.924362,-8.744800,3.174346,-6.114885,7.961921,-6.031556,0.335317,-0.311965,5.401508,5.221871,-9.707610,7.993547,2.493197,-0.439247,-9.197205,1.488188,-3.455764,3.823109,6.618245,-3.679385,-4.999913,-6.757058,5.505529,-7.298797,2.626323,8.725134,-4.917227,-2.144352,4.950894,-6.348847,-8.112217,5.408298,5.593891,-0.543233,6.629222,-8.587358,7.611182,0.060954,1.261747,-9.089687,0.921573,9.867604,-7.236226,8.740877,-4.342607,-2.835819,9.536747,-0.719691,-7.222353,9.909934,9.129024,9.684310,-5.485101,4.736384,0.204846,7.688828,-3.201847,-9.033289,-1.599800,-1.141589,9.191700,6.503459,4.343451,-6.973407,-0.883375,1.579206,3.517415,-6.139406,7.633597,2.791726,8.108201,4.950528,6.544771,3.381080,3.504950,5.587734,-3.402951,-3.507466,-8.258248,8.040021,-7.206140,-6.995737,-3.591028,-8.885567,7.838880,-3.576146,-7.748730,-5.272578,8.292937,7.299445,0.416617,2.533009,9.554343,8.076321,-0.749174,-5.409662,9.219002,-6.065020,8.117655,8.294340,1.311641,-5.659970,0.606978,4.159105,-9.394585,-8.094171,4.401002,4.099894,3.079885,-0.929694,1.293094,-3.215121,-8.354921,-0.555692,6.680323,-8.706948,6.164795,-6.640885,9.216354,7.020914,-9.824702,2.533142,-2.503047,-9.760177,-1.658145,7.839722,4.381237,-7.385845,2.456098,-8.312391,4.182960,4.525021,2.848019,6.515480,1.091731,-4.709795,-4.421205,-4.035223,-5.071431,-4.155469,9.927078,-3.253388,-0.463558,-0.153527,-8.803164,-8.329506,4.150991,-0.499660,-9.853620,4.955491,3.720254,1.406293,-0.081699,-2.047325,6.613454,2.448183,-9.055637,0.397370,-5.998107,-6.817132,-9.363393,-4.084007,-4.500357,-7.204444,9.738469,2.299764,6.835758,-4.984424,-8.984816,-8.040837,-6.075674,-4.529974,-8.266144,8.496841,-4.839353,4.394273,-7.707553,-9.989616,-2.943895,7.674921,2.889966,-8.211140,-3.690633,-8.734743,4.305953,-5.616029,-4.849962,9.694146,-0.687947,8.359125,-4.188414,-5.501787,8.466016,-2.642895,4.404180,0.788674,-8.459227,-7.822514,-7.200378,1.030369,6.725065,5.532000,-7.265763,-7.653658,8.755024,5.756109,-7.149632,6.002989,3.718219,-3.652634,6.030023,0.194274,5.836568,-9.501108,-3.363591,-3.217333,1.045232,7.661072,-2.704783,6.237160,1.973164,3.041317,-1.209277,8.071911,-7.861584,-8.545932,3.765538,1.391088,3.540238,3.072596,1.651222,-5.035406,9.205039,7.265706,7.745916,-5.828877,4.832088,9.508357,-1.420112,1.704522,-8.438368,-5.996463,1.479390,-6.902784,-9.848746,-2.596782,9.124547,-6.287978,-8.903584,-0.078915,-8.649677,5.145452,6.228242,0.774486,7.556234,-5.959274,5.674300,-3.453674,0.711546,6.680888,-6.453472,0.922800,-7.670768,-8.656393,9.329797,9.066939,9.655536,-9.644923,7.700933,-2.616590,-1.428427,-5.484871,-3.472427,8.142733,0.054384,0.974371,-1.265121,-1.267536,-6.414796,-1.935928,3.519394,-4.198335,-4.138296,1.188717,-4.615939,-9.591122,0.730815,-9.105992,5.166400,3.305673,8.191363,-0.555614,7.580675,-7.414752,-6.463302,-4.680446,-3.541529,-4.635315,9.949233,-3.986366,-1.556066,-6.195835,5.468279,5.246326,-4.676033,-5.715209,7.323438,-9.810023,0.618931,-5.021540,0.913214,4.118673,2.463954,-7.444159,-5.959953,6.700751,1.298601,-1.153764,-1.052400,-4.507089,-3.860064,-3.983747,6.674216,-5.867149,-8.639667,-3.906916,4.545273,4.059704,-7.648792,4.106272,8.111740,2.991806,2.660771,5.663605,8.862000,-7.399457,-6.643770,8.198680,8.269687,-7.518318,-5.852222,3.707632,4.154965,7.631984,-3.035731,-8.787088,0.688536,3.975883,4.856225,-8.959280,-6.232833,-4.931269,-9.786796,-9.359423,1.325273,4.950023,-1.050943,-7.134961,6.109391,6.012022,-0.873567,3.779346,9.340770,-4.148052,-8.229626,5.534896,-6.364365,9.241911,5.624505,-0.430327,1.277542,-4.007671,3.924596,5.754459,-0.071790,-7.185077,-4.275483,-7.442296,-8.248654,-2.361410,-1.259641,5.511287,3.740280,4.692098,0.473801,3.929637,7.946359,-1.922332,-1.395953,8.224076,-7.401525,-9.981267,-8.730632,-3.611301,9.202182,7.640036,-0.754693,-5.646638,6.236808,-6.622275,-8.661860,2.788365,9.132903,-4.370701,6.969793,6.812611,5.132692,-6.895860,-8.787430,0.809123,5.320253,5.723921,3.811916,5.924240,8.027828,3.681881,-3.516888,8.112829,-4.743135,2.801889,-4.100090,1.336450,-9.266178,0.907737,-9.124327,4.166300,7.748405,0.793048,-5.055785,8.185964,-8.425677,6.976093,-4.158251,4.838506,-3.010689,-0.984408,7.504941,4.622916,-4.014695,-1.370453,6.949264,-0.193607,8.823185,3.233743,4.891497,1.026330,-7.697585,-2.242010,-4.292671,7.080929,-3.032492,1.275304,-9.877265,-2.212270,8.343665,0.699017,-7.815380,3.062936,-2.356874,4.619548,4.324081,6.341797,-4.291131,-5.731251,8.927509,6.875462,-7.714368,3.559561,1.166229,7.974382,3.711592,1.044584,-5.029340,7.536209,-9.794285,-1.673393,8.814142,0.862638,5.795780,-5.064354,3.743466,0.904671,-3.922091,-3.453235,-7.261317,-3.822504,3.955116,4.274149,-2.593160,-0.835104,4.085072,8.048269,-5.762072,-4.634571,-7.895884,4.722530,9.387880,3.125147,1.469649,1.445041,8.991763,3.829560,-9.923346,5.710947,2.807380,4.552598,-0.642169,-0.319240,6.952318,5.141666,3.820107,9.468832,0.854609,7.736950,6.007294,-2.634139,-0.902572,9.652645,-6.794812,-6.504505,-0.542556,-8.478646,-8.417268,7.329573,-3.145766,7.092605,2.197556,-5.673854,-5.565351,-0.850543,-5.591622,-9.940854,-4.691728,3.204131,-7.166278,4.212802,6.796495,-5.507483,-0.290333,6.859929,-9.182420,-6.408777,-3.409267,-8.901500,-4.686931,-9.050784,8.323142,-2.598991,-9.565033,-2.586308,9.318726,6.782062,9.926869,-4.454533,4.208042,-0.990760,-0.269300,-5.069329,3.673638,1.094356,2.358433,8.211390,4.481624,-3.856242,4.938572,-2.070017,-3.410199,-6.398748,5.374573,3.553015,-0.563038,-3.590748,5.385503,7.985342,7.919234,-2.702565,-0.720040,2.305119,5.917190,0.086759,-4.796924,5.642257,1.735825,3.815075,-3.392681,-0.024995,-6.626418,-0.730505,6.469237,9.064425,-7.960966,1.763013,1.136151,-3.717119,-5.118317,7.868963,-4.607757,-3.783027,-3.287091,-0.242916,-9.902662,-9.194243,-9.469082,-9.614771,-2.003056,-1.279226,-0.074894,-4.258093,8.737497,-5.302190,-7.742043,-9.991877,1.830905,6.672467,-4.629009,-2.364074,-8.257630,-9.016935,9.426909,-9.307909,-1.280194,4.410823,-3.136860,-4.312127,-6.336698,3.891029,-1.512981,8.579884,-2.425967,-0.115358,2.046383,-9.516597,4.830836,-6.402482,2.575041,-1.897034,5.667700,-0.081092,9.845508,1.399295,-5.899025,9.345714,4.949919,-8.287921,4.873724,-9.767960,8.830504,-7.080772,0.714022,-1.710005,7.994741,-3.976890,-5.990242,-8.529861,5.137724,5.903429,-2.412434,9.607973,-4.667256,-3.638514,-5.610032,-4.491729,7.555447,-1.384889,1.440907,3.791390,7.526793,3.451283,3.039271,-1.151790,-3.181943,-2.659418,-4.606698,-6.209852,6.276830,-3.215950,6.880713,7.566752,1.470049,-3.801041,-6.032392,9.284711], dtype = "float32")#candidate|5763|(1980,)|const|float32
call_5762 = relay.TupleGetItem(func_2016_call(relay.reshape(const_5763.astype('float32'), [11, 12, 15])), 0)
call_5764 = relay.TupleGetItem(func_2019_call(relay.reshape(const_5763.astype('float32'), [11, 12, 15])), 0)
func_4957_call = mod.get_global_var('func_4957')
func_4964_call = mutated_mod.get_global_var('func_4964')
var_5792 = relay.var("var_5792", dtype = "float32", shape = (792,))#candidate|5792|(792,)|var|float32
const_5793 = relay.const([-3,-1,9,-3,-9,-7,-6,5,4,-6,-10,2,6,-2,8,-10,-6,5,6,10,-8,-10,-4,6,2,5,-8,5,10,-1,-5,-9,-3,8,2,-5,-7,-5,-9,-10,-4,-4,-9,4,-4,-10,-5,-10,5,10,-6,-6,-1,3,-1,-2,8,-5,2,-8,4,7,-9,10,7,-1,-4,-7,7,4,4,5,1,-9,-1,3,-7,7,-7,9,-8,6,-7,-6,2,-6,-7,6,-1,4,2,-1,8,-4,-6,7,-5,-7,-6,9,-4,-4,-6,-2,-7,-2,4,7,-1,4,-6,-3,-6,5,3,9,9,-10,-6,-5,-9,-2,-8,4,2,9,1,-10,4,8,-9,5,-2,6,-7,1,-6,8,3,-7,10,-4,-10,7,-10,-9,1,1,-1,1,6,4,4,9,-4,4,-2,4,-5,-6,-5,5,1,9,-1,-1,-2,-10,-7,-6,5,6,8,-9,-2,-7,-10,-4,8,-2,-4,7,-8,-4,8,-7,-10,10,4,3,2,4,4,-2,1,-5,-7,-3,-1,-1,9,-2,2,8,-9,3,-3,-3,6,3,-9,4,-1,10,-6,-3,-8,5,-6,5,6,-9,6,-9,-5,-10,1,10,-3,-7,9,5,4,-7,-2,6,-2,-6,6,1,-8,-6,4,-2,-6,5,9,-5,9,3,-8,-7,3,9,9,-3,-10,7,-4,-9,9,8,-6,-9,6,-4,3,5,5,7,5,-2,-8,-8,-7,5,-3,-4,-5,-3,6,-7,9,5,7,9,-2,8,-9,2,-5,3,-9,8,1,7,9,6,-6,-1,3,-7,-10,-4,-1,-8,-3,-4,-8,-6,-4,-6,-9,-10,-5,9,10,2,-3,10,3,10,-8,6,-9,-5,-7,5,-5,-2,5,-8,-9,6,7,4], dtype = "uint16")#candidate|5793|(336,)|const|uint16
const_5794 = relay.const([6.231058,-2.232906,-1.328198,7.151307,5.433923,-8.418324,-0.585084,3.868948,6.636439,-5.984712,-8.088357,6.697550,5.140684,7.653428,-1.229782,-7.088972,-5.214424,-8.959343,-2.329199,4.493698,3.033708,0.964580,-7.624710,-7.964373,3.502927,-3.639706,-8.369712,-1.432383,-1.972277,-7.064704,6.926215,1.012686,-6.795976,-6.372935,3.268131,-4.227753], dtype = "float64")#candidate|5794|(36,)|const|float64
call_5791 = relay.TupleGetItem(func_4957_call(relay.reshape(var_5792.astype('float32'), [792,]), relay.reshape(call_5714.astype('float64'), [2, 640]), relay.reshape(const_5793.astype('uint16'), [336,]), relay.reshape(const_5794.astype('float64'), [36,]), relay.reshape(call_5762.astype('float32'), [9, 220]), ), 5)
call_5795 = relay.TupleGetItem(func_4964_call(relay.reshape(var_5792.astype('float32'), [792,]), relay.reshape(call_5714.astype('float64'), [2, 640]), relay.reshape(const_5793.astype('uint16'), [336,]), relay.reshape(const_5794.astype('float64'), [36,]), relay.reshape(call_5762.astype('float32'), [9, 220]), ), 5)
output = relay.Tuple([call_5698,call_5714,const_5715,call_5725,call_5727,var_5728,bop_5750,call_5762,const_5763,call_5791,var_5792,const_5793,const_5794,])
output2 = relay.Tuple([call_5699,call_5716,const_5715,call_5726,call_5730,var_5728,bop_5750,call_5764,const_5763,call_5795,var_5792,const_5793,const_5794,])
func_5807 = relay.Function([var_5728,var_5729,var_5749,var_5792,], output)
mod['func_5807'] = func_5807
mod = relay.transform.InferType()(mod)
var_5808 = relay.var("var_5808", dtype = "float64", shape = (441,))#candidate|5808|(441,)|var|float64
var_5809 = relay.var("var_5809", dtype = "int8", shape = (729,))#candidate|5809|(729,)|var|int8
var_5810 = relay.var("var_5810", dtype = "int8", shape = (729,))#candidate|5810|(729,)|var|int8
var_5811 = relay.var("var_5811", dtype = "float32", shape = (792,))#candidate|5811|(792,)|var|float32
output = func_5807(var_5808,var_5809,var_5810,var_5811,)
func_5812 = relay.Function([var_5808,var_5809,var_5810,var_5811,], output)
mutated_mod['func_5812'] = func_5812
mutated_mod = relay.transform.InferType()(mutated_mod)
var_5874 = relay.var("var_5874", dtype = "uint16", shape = (11, 12, 1))#candidate|5874|(11, 12, 1)|var|uint16
var_5875 = relay.var("var_5875", dtype = "uint16", shape = (11, 12, 6))#candidate|5875|(11, 12, 6)|var|uint16
bop_5876 = relay.not_equal(var_5874.astype('bool'), var_5875.astype('bool')) # shape=(11, 12, 6)
func_5143_call = mod.get_global_var('func_5143')
func_5144_call = mutated_mod.get_global_var('func_5144')
call_5884 = relay.TupleGetItem(func_5143_call(), 1)
call_5885 = relay.TupleGetItem(func_5144_call(), 1)
uop_5905 = relay.tan(var_5875.astype('float64')) # shape=(11, 12, 6)
output = relay.Tuple([bop_5876,call_5884,uop_5905,])
output2 = relay.Tuple([bop_5876,call_5885,uop_5905,])
func_5908 = relay.Function([var_5874,var_5875,], output)
mod['func_5908'] = func_5908
mod = relay.transform.InferType()(mod)
mutated_mod['func_5908'] = func_5908
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5908_call = mutated_mod.get_global_var('func_5908')
var_5910 = relay.var("var_5910", dtype = "uint16", shape = (11, 12, 1))#candidate|5910|(11, 12, 1)|var|uint16
var_5911 = relay.var("var_5911", dtype = "uint16", shape = (11, 12, 6))#candidate|5911|(11, 12, 6)|var|uint16
call_5909 = func_5908_call(var_5910,var_5911,)
output = call_5909
func_5912 = relay.Function([var_5910,var_5911,], output)
mutated_mod['func_5912'] = func_5912
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5642_call = mod.get_global_var('func_5642')
func_5644_call = mutated_mod.get_global_var('func_5644')
call_5933 = relay.TupleGetItem(func_5642_call(), 0)
call_5934 = relay.TupleGetItem(func_5644_call(), 0)
output = call_5933
output2 = call_5934
func_5935 = relay.Function([], output)
mod['func_5935'] = func_5935
mod = relay.transform.InferType()(mod)
output = func_5935()
func_5936 = relay.Function([], output)
mutated_mod['func_5936'] = func_5936
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5935_call = mod.get_global_var('func_5935')
func_5936_call = mutated_mod.get_global_var('func_5936')
call_5948 = func_5935_call()
call_5949 = func_5935_call()
output = call_5948
output2 = call_5949
func_5968 = relay.Function([], output)
mod['func_5968'] = func_5968
mod = relay.transform.InferType()(mod)
mutated_mod['func_5968'] = func_5968
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5968_call = mutated_mod.get_global_var('func_5968')
call_5969 = func_5968_call()
output = call_5969
func_5970 = relay.Function([], output)
mutated_mod['func_5970'] = func_5970
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5968_call = mod.get_global_var('func_5968')
func_5970_call = mutated_mod.get_global_var('func_5970')
call_6068 = func_5968_call()
call_6069 = func_5968_call()
output = call_6068
output2 = call_6069
func_6088 = relay.Function([], output)
mod['func_6088'] = func_6088
mod = relay.transform.InferType()(mod)
mutated_mod['func_6088'] = func_6088
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6088_call = mutated_mod.get_global_var('func_6088')
call_6089 = func_6088_call()
output = call_6089
func_6090 = relay.Function([], output)
mutated_mod['func_6090'] = func_6090
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6088_call = mod.get_global_var('func_6088')
func_6090_call = mutated_mod.get_global_var('func_6090')
call_6112 = func_6088_call()
call_6113 = func_6088_call()
output = call_6112
output2 = call_6113
func_6146 = relay.Function([], output)
mod['func_6146'] = func_6146
mod = relay.transform.InferType()(mod)
mutated_mod['func_6146'] = func_6146
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6146_call = mutated_mod.get_global_var('func_6146')
call_6147 = func_6146_call()
output = call_6147
func_6148 = relay.Function([], output)
mutated_mod['func_6148'] = func_6148
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5442_call = mod.get_global_var('func_5442')
func_5444_call = mutated_mod.get_global_var('func_5444')
call_6237 = func_5442_call()
call_6238 = func_5442_call()
output = call_6237
output2 = call_6238
func_6251 = relay.Function([], output)
mod['func_6251'] = func_6251
mod = relay.transform.InferType()(mod)
mutated_mod['func_6251'] = func_6251
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6251_call = mutated_mod.get_global_var('func_6251')
call_6252 = func_6251_call()
output = call_6252
func_6253 = relay.Function([], output)
mutated_mod['func_6253'] = func_6253
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5143_call = mod.get_global_var('func_5143')
func_5144_call = mutated_mod.get_global_var('func_5144')
call_6273 = relay.TupleGetItem(func_5143_call(), 2)
call_6274 = relay.TupleGetItem(func_5144_call(), 2)
output = relay.Tuple([call_6273,])
output2 = relay.Tuple([call_6274,])
func_6277 = relay.Function([], output)
mod['func_6277'] = func_6277
mod = relay.transform.InferType()(mod)
mutated_mod['func_6277'] = func_6277
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6277_call = mutated_mod.get_global_var('func_6277')
call_6278 = func_6277_call()
output = call_6278
func_6279 = relay.Function([], output)
mutated_mod['func_6279'] = func_6279
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5200_call = mod.get_global_var('func_5200')
func_5201_call = mutated_mod.get_global_var('func_5201')
call_6299 = relay.TupleGetItem(func_5200_call(), 0)
call_6300 = relay.TupleGetItem(func_5201_call(), 0)
func_4185_call = mod.get_global_var('func_4185')
func_4190_call = mutated_mod.get_global_var('func_4190')
var_6303 = relay.var("var_6303", dtype = "float32", shape = (792,))#candidate|6303|(792,)|var|float32
var_6304 = relay.var("var_6304", dtype = "float64", shape = (1280,))#candidate|6304|(1280,)|var|float64
var_6305 = relay.var("var_6305", dtype = "uint16", shape = (336,))#candidate|6305|(336,)|var|uint16
call_6302 = relay.TupleGetItem(func_4185_call(relay.reshape(var_6303.astype('float32'), [8, 11, 9]), relay.reshape(var_6304.astype('float64'), [1280,]), relay.reshape(var_6305.astype('uint16'), [336,]), ), 4)
call_6306 = relay.TupleGetItem(func_4190_call(relay.reshape(var_6303.astype('float32'), [8, 11, 9]), relay.reshape(var_6304.astype('float64'), [1280,]), relay.reshape(var_6305.astype('uint16'), [336,]), ), 4)
func_6146_call = mod.get_global_var('func_6146')
func_6148_call = mutated_mod.get_global_var('func_6148')
call_6309 = func_6146_call()
call_6310 = func_6146_call()
output = relay.Tuple([call_6299,call_6302,var_6303,var_6304,var_6305,call_6309,])
output2 = relay.Tuple([call_6300,call_6306,var_6303,var_6304,var_6305,call_6310,])
func_6350 = relay.Function([var_6303,var_6304,var_6305,], output)
mod['func_6350'] = func_6350
mod = relay.transform.InferType()(mod)
mutated_mod['func_6350'] = func_6350
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6350_call = mutated_mod.get_global_var('func_6350')
var_6352 = relay.var("var_6352", dtype = "float32", shape = (792,))#candidate|6352|(792,)|var|float32
var_6353 = relay.var("var_6353", dtype = "float64", shape = (1280,))#candidate|6353|(1280,)|var|float64
var_6354 = relay.var("var_6354", dtype = "uint16", shape = (336,))#candidate|6354|(336,)|var|uint16
call_6351 = func_6350_call(var_6352,var_6353,var_6354,)
output = call_6351
func_6355 = relay.Function([var_6352,var_6353,var_6354,], output)
mutated_mod['func_6355'] = func_6355
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5182_call = mod.get_global_var('func_5182')
func_5184_call = mutated_mod.get_global_var('func_5184')
call_6359 = func_5182_call()
call_6360 = func_5182_call()
uop_6362 = relay.sqrt(call_6359.astype('float64')) # shape=(14, 2, 10)
uop_6364 = relay.sqrt(call_6360.astype('float64')) # shape=(14, 2, 10)
output = relay.Tuple([uop_6362,])
output2 = relay.Tuple([uop_6364,])
func_6366 = relay.Function([], output)
mod['func_6366'] = func_6366
mod = relay.transform.InferType()(mod)
mutated_mod['func_6366'] = func_6366
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6366_call = mutated_mod.get_global_var('func_6366')
call_6367 = func_6366_call()
output = call_6367
func_6368 = relay.Function([], output)
mutated_mod['func_6368'] = func_6368
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5416_call = mod.get_global_var('func_5416')
func_5418_call = mutated_mod.get_global_var('func_5418')
call_6406 = relay.TupleGetItem(func_5416_call(), 0)
call_6407 = relay.TupleGetItem(func_5418_call(), 0)
func_4590_call = mod.get_global_var('func_4590')
func_4593_call = mutated_mod.get_global_var('func_4593')
var_6409 = relay.var("var_6409", dtype = "float64", shape = (546,))#candidate|6409|(546,)|var|float64
call_6408 = relay.TupleGetItem(func_4590_call(relay.reshape(var_6409.astype('float64'), [3, 13, 14]), relay.reshape(var_6409.astype('float64'), [3, 13, 14]), ), 0)
call_6410 = relay.TupleGetItem(func_4593_call(relay.reshape(var_6409.astype('float64'), [3, 13, 14]), relay.reshape(var_6409.astype('float64'), [3, 13, 14]), ), 0)
func_4105_call = mod.get_global_var('func_4105')
func_4109_call = mutated_mod.get_global_var('func_4109')
const_6414 = relay.const([[-7.112824,-9.127816],[-5.372691,0.863133],[-2.783743,3.129412],[-9.799035,6.341707],[9.138457,0.137580],[1.094673,4.288159],[8.493893,-2.665182],[-0.135100,-0.543246],[-2.041821,-2.197304],[-2.047544,-6.858445],[-8.614457,3.311804],[8.309878,-8.074271],[-6.046820,9.902450],[-8.129772,-3.866447],[5.571599,0.025310],[7.374675,-0.173764],[-3.437983,-8.990135],[-9.259624,-6.281296]], dtype = "float64")#candidate|6414|(18, 2)|const|float64
call_6413 = func_4105_call(relay.reshape(const_6414.astype('float64'), [12, 3, 1]), relay.reshape(const_6414.astype('float64'), [12, 3, 1]), )
call_6415 = func_4105_call(relay.reshape(const_6414.astype('float64'), [12, 3, 1]), relay.reshape(const_6414.astype('float64'), [12, 3, 1]), )
func_2996_call = mod.get_global_var('func_2996')
func_3000_call = mutated_mod.get_global_var('func_3000')
const_6419 = relay.const([2.121074,-8.064433,-7.862447,0.897228,-4.642412,9.078967,-8.613502,-9.455161,-0.483484,6.519109,-6.382218,4.104231,3.843963,-5.080530,-4.865061,-5.418603,0.899497,8.501303,8.735075,-3.230949,-9.251693,-8.141629,9.522626,-7.771558,-8.678588,-6.981346,-1.877154,1.463765,8.176949,-2.288026,6.090582,8.825703,4.819110,-1.400458,3.962859,-8.603772,-1.426739,-3.622928,4.438666,0.350522,4.989485,4.373649,9.841809,1.251965,-1.944184,4.832576,2.371009,-1.541466,9.424678,8.102770,0.151796,2.816734,7.423129,8.084796,-8.167912,-7.164270,9.199570,9.939864,-5.039079,9.594854,7.628051,-0.976939,2.150661,1.421703,8.529388,3.366095,7.167878,-7.400000,-7.727417,3.629746,-1.977982,-4.704430,8.408767,0.504263,7.072938,-5.279413,9.485877,3.880012,-2.850698,8.851631,0.097295,-1.938827,7.135633,4.809312,3.513947,-8.405027,-5.915607,4.125872,1.554888,7.506854,9.577940,5.539834,-2.952799,8.040510,6.789589,-1.725515,2.181569,5.574732,1.567606,-6.260519,-5.165210,5.228886,-2.885503,-7.686000,0.109855,-4.674777,-2.665114,0.915309,-8.922380,5.264468,8.115169,-6.706236,-4.484466,-3.922140,7.835794,-7.053255,7.646477,-2.780466,-5.483132,7.374644,7.770465,-0.563958,5.151215,-5.073729,-5.543130,-3.254865,-0.281042,-9.259904,-3.559058,-1.142814,-0.993368,7.313652,-0.813953,6.302580,2.603360,-0.026591,-8.353354,-0.379042,-3.124420,2.755624,-8.391499,0.060141,-4.395454,2.810716,-7.217015,-7.220463,9.360592,-0.473312,-4.856136,-3.716656,1.637763,-9.450336,-7.419120,3.829228,3.793386,4.645524,-1.161728,-3.110909,8.434814,-1.659616,-8.588637,-1.026473,-1.137554,6.657173,-1.979639,-1.063040,4.991210,6.615566,6.865302,-9.869919,0.179425,-4.785744,-6.471903,-5.593309,8.747357,7.357707,-4.911899,9.334347,1.300913,1.397046,-9.310110,4.002352,-7.994882,6.730627,7.154804,-1.449405,-9.116025,7.211031,-3.764855,-5.816099,-3.948734,6.620891,4.475553,8.049551,3.929028,-1.669350,0.629744,6.557163,-4.162130,2.742041,8.106557,-5.276998,9.596132,-1.417757,6.553906,2.525133,8.629269,-0.187112,9.667668,0.713519,-1.898993,-7.022813,9.116647,3.973602,1.997234,2.248439,9.682522,-4.271534,2.422458,-6.432764,9.018171,-4.834035,-7.910652,9.607514,4.860675,-6.347096,-8.711504,5.688141,8.730299,4.445711,2.511988,7.070820,-0.376680,8.578017,-2.657561,-0.518115,0.417864,-8.177550,-7.848856,8.873852,-6.054021,6.326461,7.541225,-6.908602,-8.470066,1.153647,6.443976,-2.640712,-4.621453,6.487513,5.478927,-3.803701,1.874473,-7.397617,-8.018268,5.360993,4.343159,6.865409,3.063556,2.609360,5.337041,6.467842,1.110022,-7.233679,-2.638663,4.841048,0.485576,-6.790229,-6.955431,-2.894431,4.330841,-8.037357,9.195975,4.636977,8.637047,-3.522331,0.346811,7.052575,1.989525,-3.972092,-3.902979,5.590280,2.461562,-7.946295,-7.777396,-9.281679,-6.230359,-5.774396,-3.075707,-4.425124,1.778694,-7.863776,-8.849633,1.236280,2.459804,4.983015,8.566661,8.259101,7.607550,2.257573,1.081260,2.995809,-8.737602,8.619295,3.627618,-8.102690,-5.534662,2.443029,9.672653,-7.859089,-2.100067,-6.237240,-0.703334,-6.996731,8.055858,4.201948,2.927804,-5.204408,-9.269838,-2.192391,1.177746,-4.650769,0.021590,-2.336327,1.930618,7.775964,8.422185,0.923754,1.395338,8.512924,1.868266,7.946662,4.460833,-5.221219,3.523436,8.831527,-9.569851,-9.593574,3.518523,-1.045733,7.337624,-4.189660,2.717376,-3.309107,5.932756,8.119121,-6.120974,9.807386,-6.379838,-0.990526,-5.698272,-0.210117,8.518613,-9.145749,0.950014,6.842231,-8.619439,1.838214,-5.516997,-7.233116,9.854854,-7.599811,9.438866,0.658454,-9.869077,0.480626,8.349676,8.363706,2.089132,5.505669,-5.885872,7.064617,-9.492669,-8.017116,9.456165,-8.755212,-5.448337,-0.992424,-4.788099,9.074484,7.426111,0.286839,-6.559428,-4.977580,-7.008975,-0.594201,-7.377732,4.952590,9.885547,7.491845,3.917292,9.354194,4.271886,9.801819,-2.951078,-5.943472,1.004915,-3.187280,-9.439353,1.816843,0.997225,8.916170,-6.521509,-8.221536,-0.783998,-9.433176,-3.163962,7.455792,8.770554,3.334005,5.662998,-2.312416,-5.152337,-2.705545,2.694219,4.150241,1.518679,-5.212856,-1.010328,0.571716,3.166075,-2.115760,-6.524000,7.610589,-5.270550,7.022283,-7.992477,-9.399628,-3.242861,4.349192,6.627444,-5.255455,5.241010,-3.244872,-9.096118,-6.771266,1.583646,6.269461,-8.059031,2.366145,1.912026,-2.306014,-3.585852,-9.908595,6.015624,-9.243654,2.937727,5.045894,-3.698388,-8.468570,-4.106732,-7.594667,8.629101,4.201244,-5.231943,-3.152896,7.750126,9.391000,2.013046,-7.832303,6.045353,6.797057,-0.118104,7.696949,-2.401896,7.085061,-0.342013,-3.259194,3.998953,2.973515,3.127207,7.052907,-2.432911,-5.733983,1.121677,-3.731316,-1.093573,3.642792,-3.883922,9.842220,-6.918505,0.013243,-8.400352,-3.386724,9.742816,9.076658,0.687646,3.333669,8.392847,8.841553,7.436481,1.493010,-9.841098,-8.340103,8.081155,7.460221,-5.500852,-7.965013,9.754930,-2.127123,-9.892496,9.289786,-8.559489,4.798385,-2.287003,-7.324713,-8.446854,1.170488,-2.110012,3.432194,8.101848,1.823112,-6.832644,-3.162994,-6.715834,-5.189733,-3.468580,-5.837306,9.461811,-0.627071,6.454736,0.060760,-0.604477,-0.384123,-4.272536,-5.803118,-4.825306,0.300214,5.843727,-8.411083,-5.608247,-4.280500,-8.349358,0.597522,-1.232177,-4.680146,1.833411,9.686361,-5.451816,-5.981692,-0.890604,7.452301,9.583793,8.687892,-8.250524,-8.688497,-3.843493,-7.112520,-4.239432,-8.049126,0.368781,-3.557561,-9.849255,3.823260,3.363328,4.419523,-8.182898,-3.024623,5.314784,7.894216,-9.664339,-3.838050,-1.941142,5.841404,5.928592,-6.238182,-7.577242,7.637010,-6.756171,-8.221294,-8.745627,5.220915,-8.840271,-8.807345,3.942162,9.586114,-5.305213,7.505116,-3.575248,0.916169,-4.418997,6.060959,-4.819116,-0.562577,5.440105,-3.696569,-7.826986,-9.966694,9.789843,4.658720,-7.446602,-9.098896,-6.674172,-2.144874,2.994280,-3.072063,1.363226,-4.621309,-1.263953,-3.633940,0.589283,-7.653523,7.143373,-6.629095,0.740025,7.917976,8.042484,-4.310062,3.369385,-0.219649,-0.472066,7.963876,-4.803808,2.646851,-1.880210,-6.620253,-9.484132,2.597972,-9.988596,5.950820,4.517471,3.836814,-5.126838,5.535073,-2.645739,-3.158059,-5.252206,-3.263826,-5.220193,-0.627493,-7.952784,-9.321629,8.821626,-3.031879,-5.734754,3.308506,-6.075009,-7.141494,6.101936,2.777184,2.629493,-2.045592,9.126733,7.862817,4.762016,-8.872952,-3.000200,-1.155567,-1.835976,6.321116,-6.598618,7.807503,-8.449483,-2.951303,5.378735,8.615060,9.383088,-9.221578,1.003187,-9.477227], dtype = "float64")#candidate|6419|(660,)|const|float64
var_6420 = relay.var("var_6420", dtype = "uint8", shape = ())#candidate|6420|()|var|uint8
call_6418 = relay.TupleGetItem(func_2996_call(relay.reshape(const_6419.astype('float64'), [11, 6, 10]), relay.reshape(const_6419.astype('float64'), [11, 6, 10]), relay.reshape(var_6420.astype('uint8'), []), ), 0)
call_6421 = relay.TupleGetItem(func_3000_call(relay.reshape(const_6419.astype('float64'), [11, 6, 10]), relay.reshape(const_6419.astype('float64'), [11, 6, 10]), relay.reshape(var_6420.astype('uint8'), []), ), 0)
func_4105_call = mod.get_global_var('func_4105')
func_4109_call = mutated_mod.get_global_var('func_4109')
call_6423 = func_4105_call(relay.reshape(call_6413.astype('float64'), [12, 3, 1]), relay.reshape(call_6413.astype('float64'), [12, 3, 1]), )
call_6424 = func_4105_call(relay.reshape(call_6413.astype('float64'), [12, 3, 1]), relay.reshape(call_6413.astype('float64'), [12, 3, 1]), )
var_6433 = relay.var("var_6433", dtype = "float64", shape = (546,))#candidate|6433|(546,)|var|float64
bop_6434 = relay.divide(var_6409.astype('float64'), relay.reshape(var_6433.astype('float64'), relay.shape_of(var_6409))) # shape=(546,)
func_5642_call = mod.get_global_var('func_5642')
func_5644_call = mutated_mod.get_global_var('func_5644')
call_6444 = relay.TupleGetItem(func_5642_call(), 1)
call_6445 = relay.TupleGetItem(func_5644_call(), 1)
output = relay.Tuple([call_6406,call_6408,call_6413,const_6414,call_6418,const_6419,var_6420,call_6423,bop_6434,call_6444,])
output2 = relay.Tuple([call_6407,call_6410,call_6415,const_6414,call_6421,const_6419,var_6420,call_6424,bop_6434,call_6445,])
func_6454 = relay.Function([var_6409,var_6420,var_6433,], output)
mod['func_6454'] = func_6454
mod = relay.transform.InferType()(mod)
mutated_mod['func_6454'] = func_6454
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6454_call = mutated_mod.get_global_var('func_6454')
var_6456 = relay.var("var_6456", dtype = "float64", shape = (546,))#candidate|6456|(546,)|var|float64
var_6457 = relay.var("var_6457", dtype = "uint8", shape = ())#candidate|6457|()|var|uint8
var_6458 = relay.var("var_6458", dtype = "float64", shape = (546,))#candidate|6458|(546,)|var|float64
call_6455 = func_6454_call(var_6456,var_6457,var_6458,)
output = call_6455
func_6459 = relay.Function([var_6456,var_6457,var_6458,], output)
mutated_mod['func_6459'] = func_6459
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5178_call = mod.get_global_var('func_5178')
func_5179_call = mutated_mod.get_global_var('func_5179')
call_6480 = relay.TupleGetItem(func_5178_call(), 0)
call_6481 = relay.TupleGetItem(func_5179_call(), 0)
func_6350_call = mod.get_global_var('func_6350')
func_6355_call = mutated_mod.get_global_var('func_6355')
const_6496 = relay.const([[-1.666010,4.606944,0.888107,-8.403479,-2.773194,2.643938,2.059060,-9.110373,-4.010933,-8.171765,-6.335653,4.981318,5.763651,9.548151,-7.055958,-9.659103,2.818166,1.760056,6.516268,-2.603298,-6.361716,-7.052956,1.311707,-0.598818,-2.207787,2.515424,2.293740,-8.141794,-3.000530,-9.893955,8.604488,3.878981,-0.335987,2.637885,-8.872120,-0.233259,5.587500,-1.206457,4.477467,3.508336,-7.071867,-3.336651,1.021331,-1.361130,-1.521028,-5.588699,9.263009,-2.282872,2.100365,-6.503113,6.499512,-3.664415,3.921571,4.190646,3.928279,-1.725300,-7.007622,8.228571,7.081590,1.058788,-0.935851,8.475013,9.404282,-7.564900,2.261740,-6.867047,7.669000,-2.583057,0.441245,8.495626,-8.278063,-3.941044,-3.476653,-4.238991,-9.416153,2.526995,-0.630258,-5.517996,-5.670321,-0.797394,0.393612,2.458038,3.183686,-4.791865,3.258609,4.493518,1.359691,-8.055951,8.881791,-4.647935,-7.217784,-3.140250,-3.413345,-2.994739,8.121774,6.631983,7.923051,7.279122,-9.861782,-5.452894,5.124763,5.611770,8.191856,-7.900803,-7.517839,-6.502429,-2.737524,2.145290,-9.115663,-9.060438,-6.651876,-1.395266,-9.407107,-9.458544,0.609590,-2.513078,-4.512408,-6.892752,-4.742697,-5.638586,3.381319,5.317684,-3.915919,-5.594891,6.772467,-4.865986,-7.325255,0.413926,3.902483,9.981035,-8.750756,2.296226,5.136110,8.121296,1.097895,7.873845,-0.306627,-3.425169,5.453737,2.097926,4.765391,2.372042,9.760780,-6.622658,-6.621009,4.228010,-0.224803,-9.729572,-1.599494,3.698164,4.092527,-5.496970,7.836599,-9.565855,-1.748495,9.807619,-3.902881,0.367185,-0.481707,-5.128418,-7.111651,7.241368,-2.520419,-1.550618,9.438202,-2.175850,-5.452608,7.287168,-2.392178,0.458888,4.700607,-9.012376,4.238461,7.378502,-8.069118,-7.951824,-2.025024,6.775407,6.581814,9.084627,1.011853,-8.791642,8.176610,5.513227,1.724899,-2.788482,-7.891208,-4.710252,-5.343628,6.472054,7.634712,5.381310,4.021799,3.140881,8.735785,-2.598785,-2.011156,-1.885679,-8.599592,5.911301,-3.376291,-1.068307,2.167049,-0.522019,8.263798,-6.050427,-4.399790,-3.535165,-6.111640,-8.636615,-7.677260,-6.710738,2.777485,-3.535165,-3.065158,-7.026637,5.465408,5.308926,9.268869,3.609327,9.648913,-0.267284,4.467600,-1.207146,-1.664894,9.028509,6.806077,4.738879,9.786270,-2.715185,-6.980293,-9.208579,7.508691,-4.524649,-7.219378,7.976038,6.377864,8.219719,-5.608347,-6.146072,-1.242585,-2.402829,0.018244,9.049599,5.080703,-1.859393,-3.526173,-5.614111,7.793447,-5.021986,-5.279502,6.819535,0.064331,-6.898472,-5.281402,-2.576982,3.851451,-4.719883,2.639283,1.619828,9.178073,-1.773608,-8.478864,-7.437974,2.649587,3.705876,-0.316454,4.823219,4.816253,-5.646866,-8.694327,5.084829,7.737234,-1.541532,8.961069,1.195128,0.236615,-9.875209,-3.062403,1.020510,8.282314,9.371198,-6.390997,-3.450051,9.630767,4.868442,6.680855,1.127699,5.988620,0.875860,8.809361,1.648320,-2.490303,-9.832592,-8.448351,-2.967044,9.495293,-7.610352,-6.174843,3.002983,1.341043,2.542225,8.722446,7.415652,-1.826067,8.274298,2.578554,1.398819,-8.865808,3.627184,8.535285,9.748834,6.870926,-3.216580,-1.590479,3.218860,4.811661,-0.420799,5.847275,4.136079,0.715755,9.170365,2.432175,-8.776348,1.702446,9.866972,-3.457175,5.344773,-0.058329,1.579513,8.820028,-9.391044,-1.511783,-0.525273,-2.287458,8.072405,-7.169381,-4.217504,0.188013,-6.204803,-2.569917,-8.539326,-7.673869,8.454557,4.996907,7.664496,0.286251,0.475271,-4.848042,-3.441452,5.595775,-3.264008,-9.167633,-5.154075,-0.240773,-6.885637,-4.013356,8.663173,-6.308685,1.417023,0.472418,9.865373,-1.477939,-9.411187,-8.229569,1.319352,4.659814,1.826854,-3.247419,3.872955,-6.396825,-7.071689,-7.042150,9.888279,-9.614201,5.005960,4.644084,-2.864783,8.582031,9.977817,8.251547,-3.178607,-8.658544,6.646618,8.042224,5.327590,-0.358677,-1.244130,0.448579,-5.983859,-9.453792,-3.966799,-6.392370,8.784964,6.054107,7.380942,6.885961,4.200398,2.242194,-5.811934,-5.990378,3.029519,7.942674,2.556808,3.590907,-1.185775,1.083999,-8.064482,-1.509087,-6.654963,-5.474109,-6.034529,2.603979,9.915137,-5.723049,4.389875,-6.867875,-0.923349,-4.685717,-0.558292,8.834881,3.142125,0.337684,-8.612263,7.775639,-4.325371,-3.265620,9.634521,-4.248771,-7.993101,-8.017572,3.160559,-4.427331,2.371304,-2.430245,7.962015,-2.302331,-8.825428,-9.211423,7.473325,3.392229,5.775315,2.837506,7.132632,6.418315,-8.893298,9.529927,-1.484005,8.412480,-5.988791,-4.710196,8.003914,-7.844298,-9.322426,-6.782820,-9.373924,3.698575,5.339230,-5.676860,9.362493,5.523369,-3.860689,9.110281,-3.638004,-1.571953,-7.322195,-8.303459,8.890748,-3.174778,6.167497,-5.591087,-3.534975,0.690113,-9.127479,2.431676,-0.342670,7.817076,3.942076,6.055381,-7.672069,6.819651,-9.805489,-1.770888,-2.184297,-1.395526,-8.260882,-4.752993,-3.639264,-1.570272,6.438376,-0.344842,-5.710184,-1.967724,3.471121,-5.866740,1.186386,-7.229959,3.398685,-0.759373,1.402401,-8.100810,7.447052,4.639451,-8.977707,-8.968933,9.482215,2.017605,-8.766682,7.987007,-8.105004,9.374924,1.933313,0.311072,2.103952,7.516934,-8.681232,-7.499818,-7.771203,-9.620298,6.200299,-4.052818,1.595057,1.029556,-7.095906,3.923172,-7.759948,-9.500509,0.597182,-8.609380,1.052733,4.760011,6.755693,7.642297,4.370166,6.817804,-4.387957,3.109656,3.450573,-9.517860,-3.025564,-9.381579,-7.288282,-3.560310,3.720583,-4.447444,-5.112488,-9.616045,8.881772,9.166429,0.464057,-5.489832,2.182456,-2.476854,-7.820355,-7.640045,-1.391437,1.617631,0.666197,-0.534644,-7.689163,-7.253142,5.648045,6.379672,-4.273814,4.986152,5.469090,-8.004789,-8.636224,5.188136,5.307622,4.473763,1.237996,-3.141824,-7.222359,5.156261,-4.293071,-9.022432,-6.318765,-9.420672,-6.897263,8.363784,9.662084,-3.366882,-1.485790,-2.303720,4.791331,4.895174,-0.131063,-4.686847,9.945373,-2.862412,0.469045,3.447697,-1.066162,-2.524261,6.088478,-7.761697,9.463735,-3.596340,-7.480630,0.029539,-8.474515,-7.514670,2.446839,-3.214390,-4.277169,-7.654432,1.160922,-4.230830,3.041775,-8.062012,-3.309596,-0.766274,-4.863687,8.137411,3.277969,-2.197080,1.678863,4.558141,9.812332,1.974914,-4.734157,9.266790,2.409960,3.180555,3.365728,-7.495163,-8.525870,7.506282,6.969799,4.813903,-6.507278,5.202212,3.174283,-2.665693,-5.089912,-0.073363,5.622669,-0.019423,3.434106,8.001212,1.932054,-0.752278,-2.296658,-9.913610,-2.504062,7.624797,-9.357498,3.000220,8.138344,-8.559145,3.190536,-3.667972,5.615365,-3.315949,-6.292246,3.054209,7.509675,4.691690,7.645887,6.316833,5.857726,-0.669920,3.369851,9.016407,0.547581,-0.372265,6.686130,2.473076,0.388950,7.852498,-3.154253,-3.225095,2.336810,8.925496,6.428850,-0.938209,6.954176,-5.428739,-7.131927,1.623032,-8.067439,7.876790,-1.083843,-8.637380,-6.683770,8.813657,-0.084606,-3.425158,3.430554,-0.509256,-0.817783,-4.869794,-1.331809,0.883845,-9.501952,-5.631538,-9.699249,-1.112106,9.042328,5.672995,8.131718,-3.958816,-5.388540,6.839822,-1.157653,-3.836246,-4.508721,8.816024,-2.192990,-3.569576,6.207477,5.299259,-9.434519,2.809262,-2.282883,6.624112,3.716086,8.947597,-5.877621,-7.870955,-9.267300,5.454852,6.052169,9.035091,0.114971,-7.865545,4.223504,8.474720,-9.035857,0.564239,5.110621,-0.247523,-4.753562,-2.865901,-1.580801,6.157748,6.424570,6.129635,4.121274,-0.782466,-7.596167,-2.342808,-5.071379,7.671628,-4.210493,8.111857,6.541135,-9.261756,0.145421,-0.927844,2.372745,1.156944,5.454886,9.076623,0.131418,9.682652,5.122903,1.984915,1.163503,-0.926571,4.645989,8.097411,9.685625,-4.138474,-9.844463,-0.239169,7.458542,2.280659,-7.932876,9.928109,-1.925213,-3.960759,-4.399010,-0.896635,-5.252060,5.489891,-5.778521,8.070341,2.627586,8.065772,-2.328777,4.408052,9.391307,-8.255125,-2.899475,3.036294,6.405302,-4.014955,-4.121680,1.821500,8.173923]], dtype = "float32")#candidate|6496|(1, 792)|const|float32
var_6497 = relay.var("var_6497", dtype = "float64", shape = (64, 20))#candidate|6497|(64, 20)|var|float64
var_6498 = relay.var("var_6498", dtype = "uint16", shape = (336,))#candidate|6498|(336,)|var|uint16
call_6495 = relay.TupleGetItem(func_6350_call(relay.reshape(const_6496.astype('float32'), [792,]), relay.reshape(var_6497.astype('float64'), [1280,]), relay.reshape(var_6498.astype('uint16'), [336,]), ), 2)
call_6499 = relay.TupleGetItem(func_6355_call(relay.reshape(const_6496.astype('float32'), [792,]), relay.reshape(var_6497.astype('float64'), [1280,]), relay.reshape(var_6498.astype('uint16'), [336,]), ), 2)
output = relay.Tuple([call_6480,call_6495,const_6496,var_6497,var_6498,])
output2 = relay.Tuple([call_6481,call_6499,const_6496,var_6497,var_6498,])
func_6506 = relay.Function([var_6497,var_6498,], output)
mod['func_6506'] = func_6506
mod = relay.transform.InferType()(mod)
mutated_mod['func_6506'] = func_6506
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6506_call = mutated_mod.get_global_var('func_6506')
var_6508 = relay.var("var_6508", dtype = "float64", shape = (64, 20))#candidate|6508|(64, 20)|var|float64
var_6509 = relay.var("var_6509", dtype = "uint16", shape = (336,))#candidate|6509|(336,)|var|uint16
call_6507 = func_6506_call(var_6508,var_6509,)
output = call_6507
func_6510 = relay.Function([var_6508,var_6509,], output)
mutated_mod['func_6510'] = func_6510
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5968_call = mod.get_global_var('func_5968')
func_5970_call = mutated_mod.get_global_var('func_5970')
call_6514 = func_5968_call()
call_6515 = func_5968_call()
output = relay.Tuple([call_6514,])
output2 = relay.Tuple([call_6515,])
func_6517 = relay.Function([], output)
mod['func_6517'] = func_6517
mod = relay.transform.InferType()(mod)
mutated_mod['func_6517'] = func_6517
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6517_call = mutated_mod.get_global_var('func_6517')
call_6518 = func_6517_call()
output = call_6518
func_6519 = relay.Function([], output)
mutated_mod['func_6519'] = func_6519
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5332_call = mod.get_global_var('func_5332')
func_5333_call = mutated_mod.get_global_var('func_5333')
call_6575 = relay.TupleGetItem(func_5332_call(), 0)
call_6576 = relay.TupleGetItem(func_5333_call(), 0)
func_3139_call = mod.get_global_var('func_3139')
func_3143_call = mutated_mod.get_global_var('func_3143')
const_6627 = relay.const([5.139787,1.397077,-1.332865,7.075365,-0.171478,9.778453,-3.157530,-0.613428,-0.369965,3.900080,0.043446,3.742446,-9.618703,8.867303,-9.057132,9.031815,-6.922596,0.660521,-8.426078,-5.480953,6.412420,-7.336602,-9.888572,9.156839,6.671638,2.516578,-7.621826,8.173103,-8.342525,-2.430932,8.002529,3.616983,-7.164501,-1.340949,7.237081,7.365701,7.394688,-7.127304,-3.509135,-9.937578,-7.837527,-6.805089,-9.821284,9.355124,6.472668,8.637484,1.789484,-0.890755,-8.951303,-5.490654,7.404122,-6.433670,-4.930690,-9.750369,-5.522567,-0.782022,-7.913358,8.104311,2.366186,-6.694453,9.309778,2.897726,-9.485505,-1.085372,7.714730,-5.947550,3.422725,4.051064,0.627622,-2.134001,-9.166355,2.428667,-6.529437,-0.847487,8.889471], dtype = "float64")#candidate|6627|(75,)|const|float64
var_6628 = relay.var("var_6628", dtype = "float64", shape = (35, 15))#candidate|6628|(35, 15)|var|float64
call_6626 = relay.TupleGetItem(func_3139_call(relay.reshape(const_6627.astype('float64'), [5, 15, 1]), relay.reshape(var_6628.astype('float64'), [5, 15, 7]), ), 0)
call_6629 = relay.TupleGetItem(func_3143_call(relay.reshape(const_6627.astype('float64'), [5, 15, 1]), relay.reshape(var_6628.astype('float64'), [5, 15, 7]), ), 0)
output = relay.Tuple([call_6575,call_6626,const_6627,var_6628,])
output2 = relay.Tuple([call_6576,call_6629,const_6627,var_6628,])
func_6631 = relay.Function([var_6628,], output)
mod['func_6631'] = func_6631
mod = relay.transform.InferType()(mod)
var_6632 = relay.var("var_6632", dtype = "float64", shape = (35, 15))#candidate|6632|(35, 15)|var|float64
output = func_6631(var_6632)
func_6633 = relay.Function([var_6632], output)
mutated_mod['func_6633'] = func_6633
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6088_call = mod.get_global_var('func_6088')
func_6090_call = mutated_mod.get_global_var('func_6090')
call_6652 = func_6088_call()
call_6653 = func_6088_call()
output = relay.Tuple([call_6652,])
output2 = relay.Tuple([call_6653,])
func_6666 = relay.Function([], output)
mod['func_6666'] = func_6666
mod = relay.transform.InferType()(mod)
mutated_mod['func_6666'] = func_6666
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6666_call = mutated_mod.get_global_var('func_6666')
call_6667 = func_6666_call()
output = call_6667
func_6668 = relay.Function([], output)
mutated_mod['func_6668'] = func_6668
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6251_call = mod.get_global_var('func_6251')
func_6253_call = mutated_mod.get_global_var('func_6253')
call_6747 = func_6251_call()
call_6748 = func_6251_call()
output = call_6747
output2 = call_6748
func_6751 = relay.Function([], output)
mod['func_6751'] = func_6751
mod = relay.transform.InferType()(mod)
mutated_mod['func_6751'] = func_6751
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6751_call = mutated_mod.get_global_var('func_6751')
call_6752 = func_6751_call()
output = call_6752
func_6753 = relay.Function([], output)
mutated_mod['func_6753'] = func_6753
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5935_call = mod.get_global_var('func_5935')
func_5936_call = mutated_mod.get_global_var('func_5936')
call_6787 = func_5935_call()
call_6788 = func_5935_call()
output = relay.Tuple([call_6787,])
output2 = relay.Tuple([call_6788,])
func_6794 = relay.Function([], output)
mod['func_6794'] = func_6794
mod = relay.transform.InferType()(mod)
output = func_6794()
func_6795 = relay.Function([], output)
mutated_mod['func_6795'] = func_6795
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6088_call = mod.get_global_var('func_6088')
func_6090_call = mutated_mod.get_global_var('func_6090')
call_6804 = func_6088_call()
call_6805 = func_6088_call()
output = call_6804
output2 = call_6805
func_6816 = relay.Function([], output)
mod['func_6816'] = func_6816
mod = relay.transform.InferType()(mod)
output = func_6816()
func_6817 = relay.Function([], output)
mutated_mod['func_6817'] = func_6817
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6666_call = mod.get_global_var('func_6666')
func_6668_call = mutated_mod.get_global_var('func_6668')
call_6868 = relay.TupleGetItem(func_6666_call(), 0)
call_6869 = relay.TupleGetItem(func_6668_call(), 0)
output = call_6868
output2 = call_6869
func_6875 = relay.Function([], output)
mod['func_6875'] = func_6875
mod = relay.transform.InferType()(mod)
output = func_6875()
func_6876 = relay.Function([], output)
mutated_mod['func_6876'] = func_6876
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6794_call = mod.get_global_var('func_6794')
func_6795_call = mutated_mod.get_global_var('func_6795')
call_6877 = relay.TupleGetItem(func_6794_call(), 0)
call_6878 = relay.TupleGetItem(func_6795_call(), 0)
output = relay.Tuple([call_6877,])
output2 = relay.Tuple([call_6878,])
func_6880 = relay.Function([], output)
mod['func_6880'] = func_6880
mod = relay.transform.InferType()(mod)
output = func_6880()
func_6881 = relay.Function([], output)
mutated_mod['func_6881'] = func_6881
mutated_mod = relay.transform.InferType()(mutated_mod)
var_6934 = relay.var("var_6934", dtype = "float32", shape = (6, 15, 11))#candidate|6934|(6, 15, 11)|var|float32
uop_6935 = relay.log10(var_6934.astype('float32')) # shape=(6, 15, 11)
func_5182_call = mod.get_global_var('func_5182')
func_5184_call = mutated_mod.get_global_var('func_5184')
call_6947 = func_5182_call()
call_6948 = func_5182_call()
func_3359_call = mod.get_global_var('func_3359')
func_3361_call = mutated_mod.get_global_var('func_3361')
var_6954 = relay.var("var_6954", dtype = "float64", shape = (4, 16))#candidate|6954|(4, 16)|var|float64
call_6953 = relay.TupleGetItem(func_3359_call(relay.reshape(var_6954.astype('float64'), [2, 16, 2])), 0)
call_6955 = relay.TupleGetItem(func_3361_call(relay.reshape(var_6954.astype('float64'), [2, 16, 2])), 0)
output = relay.Tuple([uop_6935,call_6947,call_6953,var_6954,])
output2 = relay.Tuple([uop_6935,call_6948,call_6955,var_6954,])
func_6968 = relay.Function([var_6934,var_6954,], output)
mod['func_6968'] = func_6968
mod = relay.transform.InferType()(mod)
var_6969 = relay.var("var_6969", dtype = "float32", shape = (6, 15, 11))#candidate|6969|(6, 15, 11)|var|float32
var_6970 = relay.var("var_6970", dtype = "float64", shape = (4, 16))#candidate|6970|(4, 16)|var|float64
output = func_6968(var_6969,var_6970,)
func_6971 = relay.Function([var_6969,var_6970,], output)
mutated_mod['func_6971'] = func_6971
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6517_call = mod.get_global_var('func_6517')
func_6519_call = mutated_mod.get_global_var('func_6519')
call_7009 = relay.TupleGetItem(func_6517_call(), 0)
call_7010 = relay.TupleGetItem(func_6519_call(), 0)
output = relay.Tuple([call_7009,])
output2 = relay.Tuple([call_7010,])
func_7011 = relay.Function([], output)
mod['func_7011'] = func_7011
mod = relay.transform.InferType()(mod)
output = func_7011()
func_7012 = relay.Function([], output)
mutated_mod['func_7012'] = func_7012
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5178_call = mod.get_global_var('func_5178')
func_5179_call = mutated_mod.get_global_var('func_5179')
call_7131 = relay.TupleGetItem(func_5178_call(), 0)
call_7132 = relay.TupleGetItem(func_5179_call(), 0)
output = call_7131
output2 = call_7132
func_7136 = relay.Function([], output)
mod['func_7136'] = func_7136
mod = relay.transform.InferType()(mod)
output = func_7136()
func_7137 = relay.Function([], output)
mutated_mod['func_7137'] = func_7137
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6816_call = mod.get_global_var('func_6816')
func_6817_call = mutated_mod.get_global_var('func_6817')
call_7228 = func_6816_call()
call_7229 = func_6816_call()
func_5807_call = mod.get_global_var('func_5807')
func_5812_call = mutated_mod.get_global_var('func_5812')
var_7233 = relay.var("var_7233", dtype = "float64", shape = (7, 63))#candidate|7233|(7, 63)|var|float64
var_7234 = relay.var("var_7234", dtype = "int8", shape = (27, 27))#candidate|7234|(27, 27)|var|int8
const_7235 = relay.const([-8.720939,3.183868,0.826634,1.118322,4.130264,5.397279,5.588142,-7.083951,-4.138212,3.845369,6.772302,-4.479661,5.602747,6.772157,1.167841,2.698883,2.516074,1.666656,1.470785,6.901513,-4.172761,-4.398752,-7.223801,-5.373928,-5.993740,-5.565527,9.798141,5.314491,-6.030476,5.053226,-5.227163,2.623166,3.602482,-5.315675,-4.622216,9.926403,4.292672,-3.825633,-6.148494,-9.814699,-0.263434,-8.667298,4.519792,-9.608534,8.528729,5.656792,-5.174524,4.020793,3.441756,1.493775,6.359368,0.315529,-9.302877,2.407119,-8.452024,0.199327,-0.216113,-0.031034,-4.559319,2.332838,-1.474662,2.655261,1.452289,-1.062620,-2.430012,-3.013659,-9.211374,-4.528442,1.430343,9.038880,7.329970,-0.463590,4.523671,-5.638201,4.625107,7.577939,-8.079900,0.972082,6.419562,2.574528,6.793062,-2.831031,3.151557,-6.863958,-7.816786,-6.600269,3.522170,8.967202,9.914017,-8.841994,5.032364,-8.772862,-5.161871,4.868076,1.661685,0.039085,8.916668,-9.157385,3.229901,2.071464,-0.260922,7.664415,9.535127,-2.377833,-8.425240,2.596943,-6.285781,-6.243088,0.817354,9.012696,-1.870279,3.786846,-2.165047,-8.309935,9.721852,4.677555,5.961698,3.539694,2.715000,6.402230,-3.605771,2.309581,-5.874577,8.569463,8.771131,-8.118839,4.506431,0.355674,6.605362,1.662243,-5.846463,-7.637218,-9.661522,-6.712026,-0.290557,9.136523,-0.509596,6.223316,-4.750018,6.869872,6.793666,0.186767,4.876814,-4.680268,-0.559326,8.462718,-0.686804,-5.251916,8.692036,-9.372041,-4.083596,-3.911281,-1.289882,1.984325,-2.101323,7.557966,-5.104541,2.572157,8.784710,-2.096378,5.449568,-2.111134,-8.705419,-2.358462,3.923754,1.063664,4.451669,5.084910,0.826521,-3.704764,8.708074,-2.481605,8.472191,6.074216,2.929688,-2.883950,-3.933486,8.307636,-5.763252,-5.351419,-7.016890,9.613860,-8.972855,-4.871937,-6.380432,0.894457,-4.623382,5.439509,4.521161,-8.388754,0.191450,-8.331469,-2.361967,-0.383563,4.990743,-6.018751,-6.844324,5.589092,-0.364965,-1.714471,0.972009,-7.240122,6.141948,6.638402,2.559349,-7.771153,-4.293031,-8.888002,-1.889868,6.477502,8.659809,3.847801,-2.747957,-6.277256,-6.835169,7.384728,1.831702,1.524711,5.035486,4.667151,4.274803,5.760467,-5.055528,-2.517758,-7.803090,7.188413,3.267982,2.310808,2.362447,0.288765,0.643957,1.201826,-4.662721,4.746103,1.208528,-2.770650,-2.011509,6.614074,5.613418,8.660090,1.506313,-9.691160,-5.119163,-5.151014,9.912791,6.719987,-9.899437,8.199287,-9.088483,9.426415,-7.187231,-1.194397,0.299523,0.873638,6.836683,0.660907,7.518958,7.382408,5.676474,-7.652996,7.923319,6.522748,-3.055263,4.477686,-6.454518,6.784381,-7.291054,1.343432,2.346081,2.377648,-9.243357,5.827166,-5.097137,9.443232,1.150992,-6.266484,4.022602,-0.212199,2.992120,-9.246973,-6.738379,-3.430525,1.654281,4.780122,-6.586454,4.681534,0.476863,7.276059,-9.688368,8.122427,-9.289150,-5.011617,-0.101715,0.092781,-0.447877,1.507240,4.908083,-6.833515,0.956111,9.029613,-3.397105,-8.402370,-5.108290,0.221502,-3.269421,-1.285080,-7.445910,2.523406,0.928103,-9.153380,-7.815036,3.408785,6.128596,4.208049,6.693579,4.486154,0.657423,6.678588,8.725381,-2.821068,1.195114,-9.653001,-6.338202,-9.424197,-6.376297,8.559026,-3.721961,2.238355,-7.687789,8.430420,7.858921,4.448321,-8.837103,-3.776621,1.518177,-7.922605,7.476941,8.532624,-5.399502,-3.730771,-1.323328,8.035430,-5.738885,2.108693,3.942945,-3.744408,1.385380,-7.046659,-1.517810,1.870946,0.800684,-8.348288,-5.019245,-1.125501,-6.244836,1.073068,0.798905,4.237768,7.159449,-7.846319,5.200267,-8.642162,2.410076,7.715289,0.207992,6.823819,-9.026879,-6.509687,6.412022,7.207134,0.815784,0.369712,6.406902,-4.040562,0.711262,-7.724739,6.152493,9.233970,8.919788,-4.910397,7.311133,9.064969,1.484583,-1.412111,3.984699,-3.030086,-6.616897,4.686768,0.021509,7.795230,-4.781143,8.915905,-4.349322,-5.346442,8.668058,-0.749447,1.803652,-1.463654,6.820666,2.307240,-2.676512,7.233266,2.955186,4.178646,-9.552041,2.553490,-4.318277,-7.721434,0.081738,-7.220225,3.713223,1.532001,-3.496139,-4.861857,-4.662079,-7.861131,2.918215,-6.601267,-7.256412,1.088376,-9.136399,-4.876512,-5.200438,9.547772,4.028570,-7.691166,-1.884223,2.011675,-0.417495,0.098301,-0.784520,1.881273,5.420631,-8.549474,1.905918,4.839678,1.717499,-1.379864,-0.425167,7.378520,4.381272,7.383889,-3.865370,4.721083,-2.033050,2.555322,-2.967398,-1.445595,-0.666314,-6.037827,7.181685,0.579878,6.523694,-4.764807,9.145369,0.035505,7.726371,0.984866,-2.460828,-6.040422,2.162799,3.593661,6.211364,9.669586,-3.270934,2.486967,6.240707,-2.263767,1.348249,3.102386,5.341939,7.674269,7.934555,-4.575203,3.505665,-0.398763,-1.797480,2.181669,2.900234,-8.496692,-6.571806,7.795380,-6.508158,-2.623215,-3.300095,7.006951,-9.460336,5.558366,2.686608,0.157947,-3.887060,3.319791,5.020625,-7.918163,0.311606,8.848873,9.318483,-1.057768,-3.177787,4.571621,-5.736716,-0.051750,1.158994,-2.613609,2.342295,-0.071451,5.857587,2.706775,-4.218054,8.645360,-6.361633,6.158261,-3.554828,7.462617,-3.329002,7.974787,8.031473,5.757472,0.361464,5.549989,-6.705955,9.622086,8.421172,-0.411117,-6.871002,-4.558295,-5.812776,-4.686818,5.973817,-9.811392,-3.656537,-2.183463,9.288365,7.466154,0.970048,-4.998016,-7.174527,2.704089,-1.744421,8.715694,8.006554,9.123739,1.790056,-6.737634,1.426555,-3.966839,0.991606,-0.826522,-5.896484,1.828269,-0.673629,3.790756,4.634669,5.278617,0.189858,9.937404,-7.196485,-3.810056,-7.172163,-6.325861,1.978673,-5.821760,6.859379,-9.630724,-8.748718,7.969294,5.291739,3.353525,8.874878,-7.695439,2.536032,4.657496,-7.823161,8.066142,0.654797,-2.237261,-4.019036,7.166453,3.024730,1.178337,6.838835,6.058945,-9.456441,-2.822108,8.673069,-0.188404,-3.416957,-6.114247,1.379798,9.686162,1.805143,-7.747851,-6.735437,4.072431,6.676447,5.444201,1.806764,5.834217,8.388948,8.056715,-2.691294,2.586269,6.903811,8.738661,8.062851,8.231184,-5.113462,-8.815922,4.065530,7.605377,5.884676,-1.585008,0.329200,-3.217558,2.980199,4.950677,4.439703,-7.495731,-0.514755,9.151171,-6.662991,-2.539580,-8.039415,-6.142819,-2.392068,-1.173452,-5.616486,-1.065099,-6.758846,3.391240,-2.912212,3.604819,0.522890,-2.914511,4.702001,-6.427313,2.577411,4.097112,2.868101,2.720705,5.301970,3.082985,-4.935784,0.082035,-0.591789,-1.996254,1.667023,-1.648242,9.242439,-6.446550,4.268041,9.115338,6.020526,-4.874766,-8.947318,-8.182800,-1.548695,3.920726,3.788458,2.646309,8.405494,-1.749002,-5.658421,-6.082303,2.150879,-8.165744,0.830601,5.881847,-2.457606,4.260594,-0.420727,-9.675608,-9.338422,8.378353,-8.292742,2.707813,1.620696,-1.050676,4.870232,9.803846,1.540230,6.113913,8.079753,5.762612,-9.167139,-2.696562,-0.840229,3.362955,8.500769,-5.682796,-4.500437,3.265321,9.504950,2.183682,-3.508489,-7.354190,-6.127043,-2.914499,-2.903043,-3.642404,9.794129,1.657044,7.487087,-3.681717,-1.199451,0.106034,-4.078654,-5.914190,-5.897755,-9.519792,-2.227854,1.648971,-7.705980,6.110736,9.489620,-2.425535,-2.624968,3.744409,-7.044118,-4.457450,-1.716321,8.815174,6.176338,7.698328,2.750205,-5.667274,-3.343658,8.049396,-2.513379,1.234414,-4.774236,-4.914576,-8.617890,-1.051298,8.898338,6.219221,-8.796082,-0.020403,-6.484229,-8.334468,-0.375056,8.026801,-8.259909,3.086250,8.231658,-8.807074,-8.982889,3.434752,-9.363712,-8.245494,-4.785726,-2.423054,-6.436394,-8.736527,-5.891977,6.306241,-4.595065,0.792160,5.724141,1.930547,5.488642,-7.659402,-6.705844,-2.380825,9.241412,-5.830406,-2.809110,3.936401,8.816438,5.748766,-6.172148,-0.025451,-0.079813,5.823890,3.573535,-5.540097,-7.242102,-1.011525,-2.048715,-7.041244,4.060446,-7.689512,1.303441,0.618914,2.826831,-3.464313,6.196778,-7.616745,-3.588325,8.623851,-4.808629,-5.281961], dtype = "float32")#candidate|7235|(792,)|const|float32
call_7232 = relay.TupleGetItem(func_5807_call(relay.reshape(var_7233.astype('float64'), [441,]), relay.reshape(var_7234.astype('int8'), [729,]), relay.reshape(var_7234.astype('int8'), [729,]), relay.reshape(const_7235.astype('float32'), [792,]), ), 2)
call_7236 = relay.TupleGetItem(func_5812_call(relay.reshape(var_7233.astype('float64'), [441,]), relay.reshape(var_7234.astype('int8'), [729,]), relay.reshape(var_7234.astype('int8'), [729,]), relay.reshape(const_7235.astype('float32'), [792,]), ), 2)
output = relay.Tuple([call_7228,call_7232,var_7233,var_7234,const_7235,])
output2 = relay.Tuple([call_7229,call_7236,var_7233,var_7234,const_7235,])
func_7239 = relay.Function([var_7233,var_7234,], output)
mod['func_7239'] = func_7239
mod = relay.transform.InferType()(mod)
var_7240 = relay.var("var_7240", dtype = "float64", shape = (7, 63))#candidate|7240|(7, 63)|var|float64
var_7241 = relay.var("var_7241", dtype = "int8", shape = (27, 27))#candidate|7241|(27, 27)|var|int8
output = func_7239(var_7240,var_7241,)
func_7242 = relay.Function([var_7240,var_7241,], output)
mutated_mod['func_7242'] = func_7242
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6816_call = mod.get_global_var('func_6816')
func_6817_call = mutated_mod.get_global_var('func_6817')
call_7254 = func_6816_call()
call_7255 = func_6816_call()
output = relay.Tuple([call_7254,])
output2 = relay.Tuple([call_7255,])
func_7259 = relay.Function([], output)
mod['func_7259'] = func_7259
mod = relay.transform.InferType()(mod)
output = func_7259()
func_7260 = relay.Function([], output)
mutated_mod['func_7260'] = func_7260
mutated_mod = relay.transform.InferType()(mutated_mod)
const_7314 = relay.const([[[3.101439,-3.499802,8.558142,1.232405,-2.096280,7.265938,-3.429542,-3.185677,8.141891,3.070335,-5.312494],[4.778172,-1.187955,-1.462932,-0.104945,3.958388,3.484352,-6.057449,-0.399702,-1.539924,-1.757990,3.152741],[-8.580865,-6.401862,1.513704,-7.811303,7.986942,5.499254,-0.965137,-8.776320,-4.351776,-7.541433,0.209599],[-9.681359,9.647063,-7.431687,-9.741307,-7.736415,-3.664597,4.057294,-9.360199,6.849009,0.426132,8.372348],[8.708319,3.335756,3.934548,2.368472,-0.693457,1.962062,-0.246457,7.907494,1.520549,7.470736,5.753784]]], dtype = "float64")#candidate|7314|(1, 5, 11)|const|float64
uop_7315 = relay.rsqrt(const_7314.astype('float64')) # shape=(1, 5, 11)
func_5572_call = mod.get_global_var('func_5572')
func_5577_call = mutated_mod.get_global_var('func_5577')
var_7321 = relay.var("var_7321", dtype = "int8", shape = (528,))#candidate|7321|(528,)|var|int8
var_7322 = relay.var("var_7322", dtype = "float32", shape = (792,))#candidate|7322|(792,)|var|float32
const_7323 = relay.const([5,7,-7,6,3,-1,-5,6,-8,7,-7,-6,9,-4,-9,8,-8,-8,2,-9,9,-3,6,10,-10,-1,-10,-1,8,-4,10,-1,-9,-5,4,-7,-1,-7,-2,3,8,8,-1,-6,8,-6,-4,-7,5,-5,4,-1,-2,4,-9,-5,-10,-6,-6,10,2,2,-10,-10,-9,-5,9,-9,-8,7,7,6,-7,-8,5,-8,-1,-5,-7,9,2,-2,2,1,2,7,10,-9,-6,-1,-7,7,9,-3,7,-6,-6,9,8,5,-8,1,8,6,-9,4,-7,-5,3,3,-8,8,-5,-1,6,5,10,1,-6,-3,-5,-2,-8,-9,5,-10,10,8,-4,10,-4,-2,-1,-4,2,-3,-4,1,3,2,1,4,9,-1,-8,-3,-6,-1,-9,-5,4,-10,-4,-10,-1,9,-6,5,3,-10,-7,-7,9,-1,3,-1,-3,-5,4,-7,10,8,9,3,1,6,8,4,-1,6,-10,-3,10,3,-3,-10,8,-1,3,-2,-7,6,-4,-3,-5,1,5,-10,1,-8,6,9,6,-1,-2,-5,1,-6,-4,-2,-5,-6,-6,-5,2,3,-1,-1,10,6,-5,3,-4,3,-6,-9,-7,-3,-3,-1,-2,-2,5,-10,5,9,8,-10,-10,10,-7,9,4,-8,-10,2,5,5,-9,-7,10,8,7,-10,-10,-3,1,-2,3,4,1,5,1,-6,-4,5,-8,-2,1,-1,2,8,7,-10,10,-6,-7,-6,3,-9,6,3,-6,-10,4,-6,7,-2,3,-7,-4,-3,7,1,5,9,3,7,6,8,9,8,7,-2,-2,5,-3,-8,2,-10,-1,-8,10,6,1,8,8,-5,6,7,-8,3,-3,-10,2,9,1,-8,1,9,-5,-1,2,10,-5,5], dtype = "uint16")#candidate|7323|(336,)|const|uint16
const_7324 = relay.const([5,-1,-4,6,-6,1,10,-7,-3,7,-7,-3,-7,4,-4,-2,-5,-3,-3,4,-8,3,-6,-10,-3,4,7,7,-8,8,5,10,-10,-9,-3,-4,8,10,8,-3,-8,2,-4,9,-1,-3,-4,6,-5,-1,-10,1,2,9,3,4,-9,-5,7,-5,1,-7,-3,1,10,8,6,-7,1,10,-8,-5,1,9,4,-10,-4,-9,2,4,-1,1,2,-4,3,-6,8,-5,-1,-1,-10,9,-8,10,-8,1,7,-5,-9,1,-4,-3,10,5,4,-8,7,-10,-6,6,-5,5,10,-2,-10,-2,-1,9,5,-3,-2,-5,8,5,-6,-2,-7,-6,7,-7,-10,-1,6,-10,-8,1,-6,8,-2,2,1,10,8,-4,-3,-2,-1,-1,7,-7,-8,2,5,-8,10,-4,3,-10,1,8,4,-6,-4,-2,-10,-3,-5,2,6,6,9,8,5,5,-1,-9,-4,2,4,9,2,2,-9,-7,4,3,-5,-9,1,6,8,9,-2,10,-4,10,10,-5,-8,-8,-6,-10,-1,1,5,-4,-5,-5,-5,-5,-3,-4,8,-4,2,1,-2,-10,1,7,-9,7,5,2,-6,-8,1,1,-3,-8,-6,4,-1,3,4,-6,-2,-8,6,6,7,10,-5,-7,-6,3,3,2,-2,6,6,10,8,4,7,5,-3,10,-5,-8,2,-3,-3,-1,-1,-2,10,6,9,1,1,10,10,-3,10,6,-7,9,5,2,6,4,7,3,9,3,-7,-3,6,-1,-9,8,-7,5,-6,7,-2,-7,10,-4,1,8,-2,-8,9,3,-8,6,-8,7,-8,4,-7,9,-1,6,-4,-4,5,5,1,2,10,3,6,2,4,6,-8,-9,-1,3,-9,-2,-6,6,8,7,5,6,9,7,-3,-7,1,-10,-10,8,9,-4,10,4,-8,-5,-10,6,8,-7,8,2,1,-2,2,-10,3,-5,4,10,-10,-6,-6,4,-6,5,-6,-3,-6,7,8,-10,-3,-5,-6,-10,-10,1,-7,-9,7,-6,8,9,6,-5,-3,-5,-2,5,5,-6,-2,-10,3,1,-1,2,-6,-7,-5,7,7,-6,2,9,-7,-2,-6,-7,10,-9,-6,-4,2,-7,6,-6,-5,4,-6,-7,-1,-3,9,-6,-9,-2,5,-9,9,-5,1,-8,8,6,-6,-8,-9,-6,2,-9,-6,-7,-9,10,-4,5,-8,4,6,1,-2,-10,1,-3,-5,-3,-5,6,9,-4,-2,-10,1,-8,3,-8,10,10,-2,-4,5,-10,8,-4,1,-1,6,-5,5,-3,-5,-5,-9,-6,-8,-9,-8,3,6,-4,-7,2,-6,2,2,-10,-8,-2,10,7,-6,4,5,9,-7,8,8,1,9,-5,-1,-8,10,6,-7,1,-9,-9,-2,-6,3,-7,8,-7,1,7,-7,-3,-4,10,-1,-7,5,5,-4,8,-6,10,-3,10,9,8,-10,-4,1,8,-6,-2,-10,-10,-9,2,3,9,9,10,7,-3,-10,-1,-2,4,-3,8,-5,-8,7,2,-3,6,2,-9,5,-6,-10,7,-8,-1,1,-4,10,-3,-3,-8,5,8,9,6,3,10,-9,-7,-10,-7,10,9,1,-8,6,1,-8,-9,-10,3,-5,7,10,10,-4,2,8,5,-5,4,-9,-3,5,5,6,-9,1,-6,9,4,5,-9,-9,-2,7,6,-4,4,10,5,9,1,-9,4,5,1,-4,-4,-5,9,-4,-10,3,-8,-5,-3,8,-6,-10,-4,8,6,-8,-7,9,-6,3,3,-2,-1,7,-3,7,-1,10,2,3,-2,-2,-5,-4,-6,-9,-5,2,-9,-9,4,-2,2,2,-9,-6,7,9,7,-5,2,-1,-6,-2,-10,7,-6,-10,-10,4,9,-2,6,-2,-7,-7,2,6,1,-5,9,-8,1,-5,10,4,-9,-8], dtype = "int8")#candidate|7324|(729,)|const|int8
call_7320 = relay.TupleGetItem(func_5572_call(relay.reshape(var_7321.astype('int8'), [8, 6, 11]), relay.reshape(var_7322.astype('float32'), [792,]), relay.reshape(const_7323.astype('uint16'), [336,]), relay.reshape(const_7324.astype('int8'), [729,]), ), 4)
call_7325 = relay.TupleGetItem(func_5577_call(relay.reshape(var_7321.astype('int8'), [8, 6, 11]), relay.reshape(var_7322.astype('float32'), [792,]), relay.reshape(const_7323.astype('uint16'), [336,]), relay.reshape(const_7324.astype('int8'), [729,]), ), 4)
output = relay.Tuple([uop_7315,call_7320,var_7321,var_7322,const_7323,const_7324,])
output2 = relay.Tuple([uop_7315,call_7325,var_7321,var_7322,const_7323,const_7324,])
func_7330 = relay.Function([var_7321,var_7322,], output)
mod['func_7330'] = func_7330
mod = relay.transform.InferType()(mod)
var_7331 = relay.var("var_7331", dtype = "int8", shape = (528,))#candidate|7331|(528,)|var|int8
var_7332 = relay.var("var_7332", dtype = "float32", shape = (792,))#candidate|7332|(792,)|var|float32
output = func_7330(var_7331,var_7332,)
func_7333 = relay.Function([var_7331,var_7332,], output)
mutated_mod['func_7333'] = func_7333
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6875_call = mod.get_global_var('func_6875')
func_6876_call = mutated_mod.get_global_var('func_6876')
call_7386 = func_6875_call()
call_7387 = func_6875_call()
func_417_call = mod.get_global_var('func_417')
func_419_call = mutated_mod.get_global_var('func_419')
const_7407 = relay.const([-0.125271,5.567975,8.565106,1.592772,6.141799,0.691902,8.626324,-4.895453,-2.206731,0.615727,-3.897052,-1.849637,1.853815,7.980477,-6.778226,-0.418996,-0.941024,-4.391881,-2.218454,-2.007869,8.202335,9.306817,8.964453,3.493070,9.458915,1.643865,-9.677720,6.983877,-8.348274,-0.754842,6.668447,2.692869,-9.954779,3.704464,-3.521674,-9.124498,-0.294690,-3.592712,5.107410,4.579995,-1.159511,-2.513306,8.716360,-4.698073,-5.829740,-3.021060,-1.393784,-4.045533,-4.675414,6.712178,6.355099,-5.861318,5.195295,2.413559,7.172390,4.586307,1.716919,-5.049751,2.001044,-1.190940,9.510126,6.500584,-6.778195,-7.919680,7.272040,5.979385,8.453529,-3.001429,-3.774840,7.784329,6.915057,4.782154,-9.006653,4.288063,9.967223,-6.293816,-3.738303,-5.743014,0.722683,1.074343,-5.556647,2.781285,8.562370,-8.003872,1.411320,3.794133,-4.696062,4.349380,-0.597863,-4.638023,-5.713314,-9.501401,-5.245121,3.267185,2.259739,-3.221104,-0.622761,0.739743,-7.617866,9.416087,9.354290,-1.222078,-5.464962,-0.843714,-2.337132,-0.885354,2.379647,9.450166,9.073199,3.532136,-0.235639,-4.225520,7.928862,-3.171029,8.243179,9.479784,-7.385189,-5.453090,1.806662,5.446909,7.241993,-0.439869,-7.003609,-5.750569,1.383714,-9.692207,4.078192,-7.394350,-4.114290,-1.845937,-1.625252,-7.097184,0.017346,9.737876,2.474678,-0.287322,-1.886426,-9.927115,8.627309,-0.476378,-5.945706,-0.466497,6.326927,6.750176,8.824931,1.774409,5.339864,-6.476268,-4.660266,-1.031967,-7.032178,-9.366154,3.937902,0.245253,6.873586,2.374463,-8.690761,-9.112110,-0.872464,1.061329,6.582482,-2.943434,-6.452802,-0.041720,-4.576396,-2.390899,6.241946,-6.187725,9.626069,6.861268,5.218334,8.045662,-0.143671,-1.668700,2.997222,-1.141048,-9.378034,8.239804,0.153849,-8.439545,-8.556622,-5.875483,5.479191,2.012991,-4.179878,-3.674510,4.592867,8.504273,4.873551,6.271389,2.796518,5.070902,-7.151990,-6.572834,-4.715017,1.908645,-9.161081,8.999245,-3.001460,-3.848694,9.852059,0.928396,-2.829937,5.473874,-0.770440,-4.681771,-3.998119,5.310047,8.966132,2.462186,-6.684910,-8.271616,-1.903047,4.590108,-5.118325,-1.062764,-1.733445,7.696923,-8.676875,-4.912263,2.469206,2.815453,-3.457415,-1.775151,-9.150046,-8.552216,-2.354451,9.711642,-5.258265,-6.470021,6.053418,-9.364283,-4.723807,-1.702747,-3.211549,7.481008,-2.776649,-0.276405,3.471717,-5.251915,-1.619909,3.778269,3.749994,5.841144,2.512617,2.014224,-4.388761,-6.815255,0.247960,-9.264236,5.085917,8.130979,-6.750125,-0.729288,-8.558133,2.513958,9.758447,5.009001,-9.245704,-4.828566,-0.519308,-4.415408,-5.577784,5.797093,4.112076,1.022426,3.928377,-3.023207,-1.810440,-9.761832,6.914168,7.246591,-7.656076,-9.646656,-8.684643,-4.538373,-1.504401,1.111620,6.274331,-5.303258,-7.946525,4.773792,-3.234245,9.939652,-9.428780,0.415664,-0.335583,3.934774,8.019046,0.370158,-3.983124,4.431835,-3.854996,2.432718,4.756936,-8.368488,-4.561458,-9.612910,-5.853950,3.718785,0.717289,-4.074326,-8.633533,-9.506707,9.516830,4.971432,9.872500,-5.422393,0.526437,4.069645,3.576404,0.654090,1.335791,-0.329239,7.877602,-9.244626,-3.996754,-4.545010,-6.945189,7.955246,-3.351605,-1.091314,2.007330,1.884245,-0.681376,5.344788,9.394441,3.235734,-8.920996,-5.777623,-3.949586,-9.773576,7.469846,6.483305,2.202334,-4.842945,1.595447,6.001416,7.861984,0.215308,5.979683,-3.246864,2.909127,4.787973,4.073478,-8.186990,0.543780,7.681747,3.660390,-9.595289,2.044194,4.035224,6.415414,5.894057,6.357151,0.241201,5.467816,1.868784,7.122527,0.408376,-1.918024,-1.010818,-3.656287,8.492147,-7.596363,1.999689,-9.959352,-9.292324,7.617135,-7.422045,4.438637,-6.920611,0.681082,0.142030,-5.930695,-5.765612,-3.210304,-1.089087,-8.562338,2.391229,-5.616180,-0.775921,3.299761,-8.750066,-4.553607,6.320635,-1.402963,-9.128477,-6.028654,8.056077,-2.761277,4.536342,-6.542227,-2.040005,0.578963,-4.146752,-9.715856,-9.192672,0.452257,-9.490005,-9.046752,4.265445,0.004253,-1.233052,-0.366866,4.520880,0.367729,8.007213,-5.972203,-4.232505,7.596756,7.206264,0.188223,1.168292,0.178357,-1.818443,-1.192007,-7.897884,-2.458052,7.668230,-9.910639,2.527149,9.971098,-6.159634,-1.893797,-9.849073,8.820861,6.840851,2.786360,5.626624,4.969888,-9.839871,-0.242660,9.987789,-1.954428,4.494521,1.141640,-6.247292,3.236224,0.984366,-6.951626,-9.445660,9.231058,-5.840315,-6.942082,6.783327,-3.817548,5.859478,3.760388,1.075237,7.404454,-7.743749,5.230478,-9.414052,5.470810,6.820707,9.647839,4.719785,5.899388,0.190393,-8.510065,2.965679,7.467497,1.428005,-1.177431,-5.414463,2.182881,-8.272764,4.495483,7.599957,-6.716632,-7.793573,-6.622767,7.222175,-8.422023,-4.808951,8.770593,2.997002,-7.730725,5.850025,8.809743,6.154144,9.119892,4.409496,-7.162878,7.212596,9.295663,-1.943121,-0.849215,-4.876591,-2.087893,2.733751,-4.755977,-2.948057,0.606431,8.519167,9.800817,4.790980,2.813216,-3.159897,6.791517,-9.387308,3.093475,7.173312,-5.359681,-1.719704,-7.203044,4.513899,-3.902595,1.757889,6.372380,-2.389441,9.276710,7.364009,8.425445,-3.128487,-2.214493,5.040874,5.313207,3.680955,-2.312792,9.461750,9.570929,-3.428867,-8.517789,8.104850,-8.222570,4.008428,1.422631,-2.563051,-6.889835,4.909878,-5.460533,-7.539128,9.809958,8.685380,-6.957468,-4.119763,7.911660,4.500628,1.208111,-2.748955,4.591708,2.971761,-7.675150,7.960247,8.167006,9.141090,3.026010,8.755840,9.781942,6.310030,1.193868,7.379558,-4.004775,6.131526,6.255707,8.668140,3.869950,2.464311,1.834743,-3.413422,-3.929076,-0.620799,9.726666,-7.559814,-8.932261,-9.108207,-7.742898,2.168673,7.166100,-3.862351,-7.512069,3.921915,-0.717316,6.755164,4.423555,-9.373312,-2.354059,4.422045,2.541848,-4.341300,0.827861,4.283688,5.896817,-4.520594,6.186301,8.504077,5.972313,5.888109,-2.805655,0.341187,-7.862195,8.185763,-2.270442,0.818662,-2.426550,5.521536,-5.182941,1.263098,2.621771,-9.699013,-6.731129,-2.221342,-1.691328,-3.829484,9.980896,-8.795471,1.771303,-3.157518,2.875990,0.819304,2.807534,8.688986,-0.193719,6.923054,9.858700,-5.691655,-3.002769,3.406450,9.065250,0.991687,2.459429,-6.176597,9.350681,-0.616812,-5.688665,-2.853696,-7.749736,-7.833609,6.333798,6.592353,6.374345,-2.130458,-7.024226,4.648538,6.113528,-8.784561,-4.341374,-5.484009,9.317941,-2.628726,-5.840946,-3.634140,8.395522,3.083738,-3.230585,2.760356,0.771899,9.834242,0.138412,8.012452,4.105539,0.823425,-5.743814,3.420734,-5.166336,-1.779369,0.270343,0.050987,3.357453,4.400354,3.385277,1.576494,-9.104967,5.882570,8.755257,5.329906,-4.384558,7.572185,-8.124441,9.943740,-5.817757,4.653373,-3.199013,-0.786329,-6.764223,6.000611,-0.821671,-9.646425,-5.780443,-9.452993,-4.291215,-4.489610,-4.165196,9.931700,9.294194,1.151713,-7.012096,6.411084,-0.123016,-1.753980,2.927687,-3.117047,8.206410,-6.050191,-6.281909,1.950564,-5.514145,-0.061687,2.024101,7.543667,-7.301577,5.281412,-6.138940,0.860337,-9.208949,8.324806,5.018160,-4.593041,-8.623823,-5.421002,6.989815,6.231195,-6.752721,6.529311,1.206241,-8.276529,9.113670,0.384662,-9.471774,-6.030773,-6.046558,-9.002883,-1.093150,-9.447850,5.257743,1.789735,6.048274,6.033325,-3.007756,8.139299,6.979188,9.491257,9.789795,-3.614993,2.503979,2.429558,-3.293551,-1.398796,-6.820924,5.398934,1.738021,-6.855015,8.574074,-5.722135,-8.142520,-8.784576,1.660982,5.912943,-8.296645,-1.048819,-3.429358,5.080362,9.271210,5.385161,9.319695,-6.814108,-2.066949,2.372016,-8.282485,6.342342,7.428234,-1.802083,-8.585390,-6.516403,-3.646483,-8.840925,-4.012636,5.889785,5.495543,8.042744,-8.008431,-1.558382,-7.263373,-7.651082,7.381482,-1.787865,0.626510,-1.853039,-2.878762,-6.713097,2.714123,5.340448,2.039785,-5.238623,-5.564241,8.340106,2.462491,8.044192,1.633814,-5.836519,5.221894,8.144783,8.002759,-4.465031,-8.569406,-2.369469,-7.724262,7.412322,0.420789,-6.636644,3.745226,-6.125588,8.032341,3.680788,3.719178,-5.753020,-5.160882,-8.768271,5.772121,7.517600,5.776311,4.923436,4.078489,1.539299,-5.993124,-9.747337,7.578788,-0.636163,-2.031257,2.374730,2.193944,8.937881,-8.032388,-0.779519,-0.042089,9.242321,5.846531,2.989484,-8.761202,4.376630,1.480038,-7.704743,-7.423355,6.722092,-0.869294,4.482567,-1.957088,6.097817,-3.390164,-1.672344,2.597323,-1.644531,7.935282,-9.040960,2.151910,-3.649049,6.916103,5.762242,-1.237692,5.559163,-3.486692,-7.272032,2.221000,6.362919,3.203789,-8.963586,2.777604,-3.061753,6.877510,-9.555285,-5.531378,2.140895,-7.159311,-7.215429,0.012679,6.109384,-1.076676,1.378705,-7.422617,8.764888,-7.019931,-6.677509,-6.490583,-5.714541,5.586617,-1.235446,-0.143717,-7.353825,3.415462,9.609308,-6.166252,-4.585669,-4.689778,-6.305618,-8.365307,1.315929,8.594191,-0.069130,-8.948510,9.796838,-1.693774,5.889873,3.050787,9.363470,-9.463602,-3.728515,4.293572,2.942811,-9.708933,-0.070490,3.818768,1.542198,-8.335037,8.257899,4.264651,-3.352408,-9.561783,-1.072410,5.979297,-2.021905,-0.801477,-4.927301,4.424169,9.436301,3.753696,-6.801819,5.518822,-1.725228,-2.203882,-2.918902,-5.589635,9.050188,4.169730,3.263945,0.935634,-0.547853,-2.081645,7.074858,5.443636,8.778219,8.969862,8.055993,0.114436,-4.720237,8.807686,0.834319,5.288800,5.885762,-3.910534,-3.723558,-2.129778,6.676696,2.629087,7.968381,-5.989487,-6.176227,-5.283151,-0.804754,5.987094,0.077958,-9.229971,-2.251842,6.472906,-1.320968,-8.379072,5.863913,3.848415,5.807259,-6.652644,9.913806,7.171455,-8.359066,-4.570498,-7.381236,9.546425,-2.463104,-7.389350,6.381230,-1.046457,-7.524917,-0.308128,-0.875757,-3.816090,9.568243,-2.235829,2.588038,-2.929149,6.604270,3.605000,-2.672631,-2.465860,-1.863483,4.002878,-0.087008,8.244209,5.019306,8.822815,6.519898,2.648195,-0.693738,-3.890959,-4.992077,3.421033,6.441761,7.135146,7.373942,-7.993525,-8.074783,5.153958,-4.208443,6.807839,9.354670,7.565288,3.581214,8.351245,-8.856927,3.681612,-4.696571,-1.816769,6.626278,3.159153,8.746835,-1.273040,3.204847,1.679346,-7.013277,2.854974,6.987144,9.138623,-2.359649,-2.402754,-3.684507,4.011791,-0.172094,-0.403903,9.330850,-7.917220,1.355492,-1.490978,-0.120415,-6.110204,1.838371,-9.276721,-7.946914,6.252838,-1.133664,7.857876,2.177167,6.120826,4.851746,-5.540233,1.175353,0.622991,3.666879,-8.076427,4.879307,0.888905,-2.115563,9.566555,7.943887,6.341366,-4.341159,-1.910740,4.119370,3.218519,6.392343,6.294737,-0.151829,3.482629,7.415542,-6.588067,6.083049,0.885469,2.301930,-6.869397,1.586597,3.092830,6.091983,1.878086,9.004396,5.063162,-5.918416,-6.307025,9.870205,9.313559,0.653604,-0.924641,9.397325,-7.373842,-1.932896,5.225296,6.498845,-1.439733,-5.447091,3.890350,-0.432633,-5.952192,-7.631243,-4.879453,-4.234339,-6.795675,7.480426,-9.043953,-1.475933,5.128355,-7.988818,-2.633703,5.096890,-4.219225,-7.887188,-4.706720,-9.304238,-4.985977,9.831132,-1.576228,-6.713396,5.400254,9.082501,-5.685916,-2.503248,3.527824,-3.226480,-0.935596,-4.172506,1.044736,1.562002,2.532753,-2.022943,-4.900805,-7.946482,3.572792,5.577530,-4.182679,4.358172,-0.963764,3.436346,-8.829945,8.532381,6.425983,3.255702,-1.272797,-0.236233,-4.137040,6.301879,-7.445829,7.862610,-4.263545,-0.299707,-0.888969,3.567702,-0.362792,-4.914292,-5.955303,7.267789,6.070919,-9.794073,7.305902,0.853018,2.504710,-3.696826,8.268406,-6.120809,-9.219108,-5.297731,-2.548092,6.632412,-9.776832,4.098619,9.466443,1.489049,8.526234,2.334009,-8.027722,-0.022794,9.123032,-4.834241,-2.391848,7.789097,4.752819,9.618409,2.180610,-7.480739,-6.771883,8.590991,7.328694,-5.014609,-5.732374,9.566718,6.956940,4.931680,9.336050,3.342579,-3.815089,-1.326018,-4.035050,7.454456,0.710074,-4.130331,8.744244,4.773750,-3.182143,1.272387,9.737896,1.305530,-8.059582,-6.963755,-8.870253,3.384214,-7.701687,2.083753,-1.334924,-0.174027,-5.408317,2.551377,-3.537040,-5.539309,0.524025,-3.119561,3.146019,4.345917,-1.210978,9.904095,8.243185,3.403944,-5.831110,-8.718342,5.438183,-3.875591,3.422140,5.989089,6.150313,-2.905918,7.415290,3.243442,2.338991,-5.206182,-7.418813,-9.971587,-6.222119,5.393189,-3.332231,7.455772,6.003793,-3.402578,-3.288671,-5.634935,8.368168,-5.287110,5.544597,8.417779,-6.795317,-5.214482,2.235379,-3.506057,-2.018802,8.904137,2.755206,-8.305192,-2.725835,-5.077446,-3.680322,-5.209123,3.403876,-2.088485,5.867120,1.910729,4.597499,-4.475592,9.332068,5.081568,2.260541,-4.252218,1.460449,2.422076,4.801299,-8.379494,-5.035933,-2.359706,7.114487,1.188624,-8.465174,-1.459528,-2.431655,7.041705,8.857794,-3.405214,1.573014,-0.022574], dtype = "float64")#candidate|7407|(1280,)|const|float64
call_7406 = func_417_call(relay.reshape(const_7407.astype('float64'), [8, 10, 16]))
call_7408 = func_417_call(relay.reshape(const_7407.astype('float64'), [8, 10, 16]))
var_7423 = relay.var("var_7423", dtype = "float64", shape = (8, 10, 16))#candidate|7423|(8, 10, 16)|var|float64
bop_7424 = relay.divide(call_7406.astype('float32'), relay.reshape(var_7423.astype('float32'), relay.shape_of(call_7406))) # shape=(8, 10, 16)
bop_7427 = relay.divide(call_7408.astype('float32'), relay.reshape(var_7423.astype('float32'), relay.shape_of(call_7408))) # shape=(8, 10, 16)
output = relay.Tuple([call_7386,const_7407,bop_7424,])
output2 = relay.Tuple([call_7387,const_7407,bop_7427,])
func_7442 = relay.Function([var_7423,], output)
mod['func_7442'] = func_7442
mod = relay.transform.InferType()(mod)
mutated_mod['func_7442'] = func_7442
mutated_mod = relay.transform.InferType()(mutated_mod)
var_7443 = relay.var("var_7443", dtype = "float64", shape = (8, 10, 16))#candidate|7443|(8, 10, 16)|var|float64
func_7442_call = mutated_mod.get_global_var('func_7442')
call_7444 = func_7442_call(var_7443)
output = call_7444
func_7445 = relay.Function([var_7443], output)
mutated_mod['func_7445'] = func_7445
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6517_call = mod.get_global_var('func_6517')
func_6519_call = mutated_mod.get_global_var('func_6519')
call_7456 = relay.TupleGetItem(func_6517_call(), 0)
call_7457 = relay.TupleGetItem(func_6519_call(), 0)
output = relay.Tuple([call_7456,])
output2 = relay.Tuple([call_7457,])
func_7461 = relay.Function([], output)
mod['func_7461'] = func_7461
mod = relay.transform.InferType()(mod)
mutated_mod['func_7461'] = func_7461
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7461_call = mutated_mod.get_global_var('func_7461')
call_7462 = func_7461_call()
output = call_7462
func_7463 = relay.Function([], output)
mutated_mod['func_7463'] = func_7463
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7461_call = mod.get_global_var('func_7461')
func_7463_call = mutated_mod.get_global_var('func_7463')
call_7468 = relay.TupleGetItem(func_7461_call(), 0)
call_7469 = relay.TupleGetItem(func_7463_call(), 0)
func_5513_call = mod.get_global_var('func_5513')
func_5516_call = mutated_mod.get_global_var('func_5516')
const_7474 = relay.const([-5.253225,-8.767737,-7.949704,-9.942958,2.850371,6.404645,-0.696970,-9.573217,-1.196823,-4.187905,-9.856054,4.226775,-7.534618,3.317378,-1.048955,7.258065,5.813409,-7.589647,-3.030863,-0.391907,-7.703337,-9.420158,0.354762,-5.929747,5.386823,8.830091,4.224451,7.815319,8.095962,-8.063276,-4.485913,8.290834,1.313937,7.772800,-3.064144,3.198944,-1.421061,4.942642,-7.370986,8.634598,5.689878,3.634411,1.768038,-9.059660,-4.282312,-1.611353,-2.029927,-4.761937,5.564904,-1.041642,2.354137,-0.382551,7.883699,2.966388,-4.903566,-9.852418,3.080900,-6.358065,-8.948887,-6.587927,1.772084,-3.750214,-2.578067,-8.120584,-7.654238,1.519235,-4.690018,8.619914,0.273112,-3.946032,3.913861,2.783954,-5.955256,3.116527,2.616317,9.885449,0.627779,-5.526950,-8.865649,4.727867,6.036567,8.344753,8.628619,-1.649232,1.862090,-2.205288,2.391158,-2.410038,0.693339,-1.134682,3.729614,-1.156228,3.283056,-6.517308,5.488172,9.968378,-5.630890,-4.305253,2.511596,-0.527249,2.738103,-0.309243,6.013191,2.481025,-1.166193,6.865447,4.513492,-9.408074,-8.229874,5.283026,5.166765,7.552524,3.355504,0.801718,-3.288077,9.334499,-2.686911,-5.217444,-1.792473,-4.901682,1.813390,-7.104000,2.046137,-4.616069,8.931589,2.605511,-6.865105,-9.921846,7.239651,-3.661173,-7.652700,4.194548,6.291773,-4.454216,8.795191,7.262430,-8.978583,-3.074494,-2.292269,4.978632,-4.960531,-6.939205,9.520477,-3.036166,-6.616750,-3.862952,-6.337263,1.506445,1.932199,-0.453604,7.536140,-5.612369,6.743175,3.231285,7.107741,-6.461433,-9.481027,-1.766880,2.244843,-0.772391,7.021656,-0.459152,-4.690717,-3.952933,0.104742,0.440458,-7.687311,-2.067567,-8.298923,3.828399,7.469994,9.495611,7.478158,2.549610,-0.251258,-1.799274,8.645379,9.168933,0.149632,6.588040,-7.988279,3.570127,-7.694886,7.601894,3.695260,-5.261120,-7.913457,-0.857218,6.615234,-2.191473,6.636337,-0.261541,-8.967232,4.806940,7.530213,-7.477098,6.740720,-0.603351,-4.928153,-1.200040,-5.832551,-5.206671,-5.423415,-1.822437,-3.707689,4.293128,-5.098727,-2.644893,0.043811,-9.916867,6.778844,2.494723,-9.579804,-8.048266,7.476586,-3.804757,-1.263765,-0.682772,-8.559524,-7.661389,4.622593,-5.787500,6.325647,-3.004583,-9.887854,0.026224,-8.149821,-6.223583,-8.856212,1.422273,5.285637,-2.626098,-9.698147,-1.271708,5.625401,7.238394,-5.319217,-9.806588,6.206750,2.417794,0.820712,2.843014,5.665577,-8.221936,-7.542496,2.456454,-4.652131,9.881033,-0.330206,0.145449,8.122080,-4.702614,-9.274780,5.356669,-7.295358,-1.697668,-0.852229,2.399946,7.076726,2.404795,9.547551,-7.045341,-9.954542,-8.621474,6.882210,7.464744,-0.811386,5.937846,-2.685780,-7.890712,7.528639,2.602932,-8.607423,4.465973,-5.839454,-2.007826,8.202024,4.882325,-3.408136,-5.546316,1.688150,-8.016747,1.390623,1.562600,9.466258,-8.491087,-4.731026,-3.152269,-9.027931,-5.600761,-1.921878,-0.115420,7.424129,-8.343272,0.365537,-7.952894,-5.989640,8.567768,-4.908183,9.947191,-8.511813,-6.766965,8.275670,-3.527660,-9.751231,5.183716,0.846179,-3.479962,2.740961,-4.844792,9.983391,-6.558629,-9.720708,2.039320,-0.487669,-9.994116,3.169658,3.331447,1.492010,-7.549203,-8.445267,-3.910853,-1.440309,6.650578,4.883251,-4.424885,-5.468773,-7.286711,-1.274729,9.602754,2.999644,-1.292348,-1.324999,0.502091,6.967126,9.410696,9.764518,6.695532,-5.897974,-6.974824,-2.298781,-1.886506,6.621806,3.659110,-5.714436,-1.398015,-9.841635,-0.068072,-3.528787,-7.498772,-2.657912,2.369102,-4.192609,8.084822,9.787084,-4.822870,-0.290001,-5.023783,-4.484352,-4.388125,-9.846433,-6.198578,1.404990,7.930588,-1.900186,4.937743,9.746176,6.638008,2.676796,-9.793334,-8.617748,-4.609329,6.550284,-5.727656,6.442414,3.940832,2.996706,-2.145354,-3.997931,-4.597922,-0.852899,5.806706,-7.663395,2.217952,1.189463,5.760936,-4.098734,3.929240,-8.576748,9.075984,-9.027692,-8.729681,-7.352868,-4.105198,-9.061514,-8.682438,-4.036273,4.828897,-7.952621,-8.512197,-3.378657,1.509144,4.328705,5.249471,-1.179021,9.350542,7.598605,-4.228827,9.395641,-9.906181,1.927771,-3.143330,8.586399,-1.582346,-9.198543,4.119372,-9.951896,3.344351,7.920560,-7.629906,-7.540543,-3.555825,3.560971,5.973686,2.645173,-5.053834,-9.844039,-3.772799,-3.941641,4.825990,1.843344,-1.990240,1.670245,-3.191609,-7.574087,-0.089561,-1.717356,-5.076468,3.482940,-9.274818,2.665165,1.288679,0.116636,3.804229,7.002519,-2.834915,-8.124009,-4.889401,0.471882,-8.260767,0.841654,-1.536177,-5.764845,9.488121,-0.225379,-2.609256,5.091813,-5.403295,-7.890650,-4.805251,-1.731776,-5.421043,5.035273,3.435357,0.606575,-3.371055,-1.094628,1.778001,7.063244,4.001908,-7.392419,-4.238422,7.152219,5.412360,-4.156430,5.287312,-6.529943,6.247040,0.313403,-1.667072,2.601593,3.440362,2.617813,-4.473198,-0.828029,8.715982,2.890772,-4.129868,-9.008496,-5.416075,4.702711,4.011252,-5.706816,-2.495641,9.618668,-7.762141,0.775344,4.459131,-5.950760,-4.998205,0.192784,-0.972746,5.557784,2.141662,-1.354481,-6.695169,-0.721722,6.274706,1.380973,-3.672811,7.174638,7.034858,-9.233541,7.121472,-4.754516,7.569057,0.351704,-3.956883,-3.923474,-9.997716,-0.723312,0.672410,6.770958,8.987898,-0.730676,4.391013,-2.084815,-2.884565,9.225402,-2.068287,2.376664,5.497649,6.390916,1.589413,-8.333961,-4.026138,6.646532,8.850274,-4.371839,8.132493,-9.706061,-7.560854,4.625724,5.571135,0.498755,-0.644685,-5.555173,9.136810,4.990610,7.061687,5.771565,-8.414868,5.861357,0.786908,-4.492520,-6.952497,2.363246,-4.887962,-9.394756,-4.493184,4.617156,-6.877971,-0.254297,0.746894,1.977458,4.115119,-6.332785,8.483448,9.339598,7.656547,6.920045,-1.694013,8.050100,5.833880,8.775524,8.990190,-9.999156,-0.674503,9.437341,-4.056697,-2.397982,5.400043,-9.716312,2.197611,0.306194,2.378078,-5.069531,-9.082916,-1.931899,6.798893,8.501669,-1.813272,1.208374,-7.073668,-6.720196,2.864203,3.422380,1.467698,1.272714,3.106430,-5.534854,0.258846,9.195019,-9.541853,7.639579,4.870119,-7.950201,-6.683035,-5.916963,-9.340206,4.771568,2.621446,6.565122,-5.815118,4.959480,7.874700,6.687390,4.866915,-8.404439,-7.621524,6.948168,-3.745857,-8.915734,2.548359,-1.837689,0.712216,9.866772,-2.940214,-8.269784,7.677723,-2.402832,3.382890,7.937241,-9.740373,-5.326434,8.485066,-4.114297,6.469129,-5.526141,5.340048,-7.783742,6.202125,-1.826307,-2.199812,-6.961809,-5.074381,3.570501,-2.119810,-1.867341,-7.990709,-5.109326,1.792999,-6.905042,4.566992,9.474534,2.108787,8.794021,-7.301517,-9.210366,-4.201129], dtype = "float64")#candidate|7474|(660,)|const|float64
call_7473 = func_5513_call(relay.reshape(const_7474.astype('float64'), [660,]))
call_7475 = func_5513_call(relay.reshape(const_7474.astype('float64'), [660,]))
output = relay.Tuple([call_7468,call_7473,const_7474,])
output2 = relay.Tuple([call_7469,call_7475,const_7474,])
func_7480 = relay.Function([], output)
mod['func_7480'] = func_7480
mod = relay.transform.InferType()(mod)
output = func_7480()
func_7481 = relay.Function([], output)
mutated_mod['func_7481'] = func_7481
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6875_call = mod.get_global_var('func_6875')
func_6876_call = mutated_mod.get_global_var('func_6876')
call_7537 = func_6875_call()
call_7538 = func_6875_call()
func_4590_call = mod.get_global_var('func_4590')
func_4593_call = mutated_mod.get_global_var('func_4593')
var_7542 = relay.var("var_7542", dtype = "float64", shape = (546,))#candidate|7542|(546,)|var|float64
call_7541 = relay.TupleGetItem(func_4590_call(relay.reshape(var_7542.astype('float64'), [3, 13, 14]), relay.reshape(var_7542.astype('float64'), [3, 13, 14]), ), 0)
call_7543 = relay.TupleGetItem(func_4593_call(relay.reshape(var_7542.astype('float64'), [3, 13, 14]), relay.reshape(var_7542.astype('float64'), [3, 13, 14]), ), 0)
output = relay.Tuple([call_7537,call_7541,var_7542,])
output2 = relay.Tuple([call_7538,call_7543,var_7542,])
func_7547 = relay.Function([var_7542,], output)
mod['func_7547'] = func_7547
mod = relay.transform.InferType()(mod)
var_7548 = relay.var("var_7548", dtype = "float64", shape = (546,))#candidate|7548|(546,)|var|float64
output = func_7547(var_7548)
func_7549 = relay.Function([var_7548], output)
mutated_mod['func_7549'] = func_7549
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6088_call = mod.get_global_var('func_6088')
func_6090_call = mutated_mod.get_global_var('func_6090')
call_7564 = func_6088_call()
call_7565 = func_6088_call()
func_3359_call = mod.get_global_var('func_3359')
func_3361_call = mutated_mod.get_global_var('func_3361')
call_7567 = relay.TupleGetItem(func_3359_call(relay.reshape(call_7564.astype('float64'), [2, 16, 2])), 0)
call_7568 = relay.TupleGetItem(func_3361_call(relay.reshape(call_7564.astype('float64'), [2, 16, 2])), 0)
func_3278_call = mod.get_global_var('func_3278')
func_3282_call = mutated_mod.get_global_var('func_3282')
var_7585 = relay.var("var_7585", dtype = "uint16", shape = (336, 1))#candidate|7585|(336, 1)|var|uint16
call_7584 = func_3278_call(relay.reshape(var_7585.astype('uint16'), [6, 8, 7]), relay.reshape(var_7585.astype('uint16'), [6, 8, 7]), )
call_7586 = func_3278_call(relay.reshape(var_7585.astype('uint16'), [6, 8, 7]), relay.reshape(var_7585.astype('uint16'), [6, 8, 7]), )
bop_7594 = relay.bitwise_xor(call_7564.astype('uint16'), var_7585.astype('uint16')) # shape=(336, 64)
bop_7597 = relay.bitwise_xor(call_7565.astype('uint16'), var_7585.astype('uint16')) # shape=(336, 64)
uop_7600 = relay.sin(var_7585.astype('float64')) # shape=(336, 1)
func_7330_call = mod.get_global_var('func_7330')
func_7333_call = mutated_mod.get_global_var('func_7333')
const_7607 = relay.const([4,6,-1,-5,6,-1,8,-4,-1,4,4,-10,9,-6,-9,-1,-4,-8,-2,5,1,-8,2,-1,-7,-5,-10,-7,-4,-3,7,-9,3,-7,10,-2,-10,9,6,-6,-6,2,-5,-4,-4,-3,-3,4,-10,10,3,-1,6,-4,4,5,-3,8,8,7,1,2,4,1,-7,6,-4,-4,8,5,9,2,7,-10,-1,6,8,-7,6,-8,10,-4,10,-7,7,10,1,10,1,-4,10,4,10,2,-1,10,4,-8,-3,5,-6,-9,-7,7,-2,10,-3,-10,10,-2,2,-2,7,3,4,1,2,-1,3,-7,-10,-5,-3,-9,-6,-4,2,-5,8,8,-7,-6,-3,8,-4,8,-3,-1,9,9,-1,-4,-1,-4,8,-10,3,1,-2,1,-5,-2,-6,-7,2,-10,1,8,2,10,3,2,6,3,4,6,7,-10,-2,7,2,-9,-1,2,10,-1,4,3,9,-9,-2,-4,2,9,10,-8,-3,-3,7,-8,-9,-1,4,-2,10,10,6,-10,-10,2,6,2,10,-6,5,5,-2,-7,-10,1,-2,-3,-7,-7,-3,6,-3,-3,-6,-10,-2,4,-1,2,2,1,-6,-2,-7,-7,10,-4,-4,5,-3,-10,2,-8,-5,-4,6,2,-2,-5,3,-3,-7,-5,-1,10,7,-10,-4,-2,3,-9,-3,-2,-3,-6,-1,9,-4,-9,-7,3,-1,-1,-5,7,8,6,-4,6,5,6,-1,-4,-1,1,5,10,-1,2,-5,-6,2,-4,-2,6,2,-3,-6,7,-1,-3,-6,-2,-5,3,-1,3,-2,-2,9,10,-7,-4,10,-10,4,7,6,7,4,-1,2,10,3,6,9,3,1,-1,-1,-3,-2,-6,-10,4,-6,10,-9,2,4,3,-6,-9,4,-7,1,-3,-2,3,-4,7,2,-10,10,4,-7,-7,-3,-9,8,-1,8,2,-3,-2,-4,8,8,8,9,-2,-10,1,-5,8,6,10,5,6,9,-6,-8,-8,2,4,9,-10,-3,2,-9,3,4,-3,-7,7,-8,-8,9,5,-1,3,-5,9,-3,-9,-3,-10,-6,-6,-2,7,-7,-6,-1,8,-10,-2,-5,4,-3,-10,3,-5,-3,-10,-5,-4,-3,9,-9,-6,9,5,-10,-1,2,-1,-3,4,9,-10,-2,-7,5,9,-2,-5,2,4,-9,-4,10,5,-4,-9,4,5,6,-3,7,-6,-8,9,-3,1,-2,2,-2,3,-8,7,4,8,-6,-2,4,2,-9,1,6,3,-8,-7,-3,-6,-5,-1,-7,-2,1,-5,-1,8,-8,3,2,9,2,2,-8,-6,-1,5,-7,2,10,4,2,3,-2,3,4,-6,8,2,6,9,5,7,1,-10,7,-2,6,1,4,4,4,6,1,7,6,-7], dtype = "int8")#candidate|7607|(528,)|const|int8
var_7608 = relay.var("var_7608", dtype = "float32", shape = (792,))#candidate|7608|(792,)|var|float32
call_7606 = relay.TupleGetItem(func_7330_call(relay.reshape(const_7607.astype('int8'), [528,]), relay.reshape(var_7608.astype('float32'), [792,]), ), 4)
call_7609 = relay.TupleGetItem(func_7333_call(relay.reshape(const_7607.astype('int8'), [528,]), relay.reshape(var_7608.astype('float32'), [792,]), ), 4)
output = relay.Tuple([call_7567,call_7584,bop_7594,uop_7600,call_7606,const_7607,var_7608,])
output2 = relay.Tuple([call_7568,call_7586,bop_7597,uop_7600,call_7609,const_7607,var_7608,])
func_7613 = relay.Function([var_7585,var_7608,], output)
mod['func_7613'] = func_7613
mod = relay.transform.InferType()(mod)
mutated_mod['func_7613'] = func_7613
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7613_call = mutated_mod.get_global_var('func_7613')
var_7615 = relay.var("var_7615", dtype = "uint16", shape = (336, 1))#candidate|7615|(336, 1)|var|uint16
var_7616 = relay.var("var_7616", dtype = "float32", shape = (792,))#candidate|7616|(792,)|var|float32
call_7614 = func_7613_call(var_7615,var_7616,)
output = call_7614
func_7617 = relay.Function([var_7615,var_7616,], output)
mutated_mod['func_7617'] = func_7617
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7011_call = mod.get_global_var('func_7011')
func_7012_call = mutated_mod.get_global_var('func_7012')
call_7647 = relay.TupleGetItem(func_7011_call(), 0)
call_7648 = relay.TupleGetItem(func_7012_call(), 0)
func_6146_call = mod.get_global_var('func_6146')
func_6148_call = mutated_mod.get_global_var('func_6148')
call_7649 = func_6146_call()
call_7650 = func_6146_call()
output = relay.Tuple([call_7647,call_7649,])
output2 = relay.Tuple([call_7648,call_7650,])
func_7656 = relay.Function([], output)
mod['func_7656'] = func_7656
mod = relay.transform.InferType()(mod)
output = func_7656()
func_7657 = relay.Function([], output)
mutated_mod['func_7657'] = func_7657
mutated_mod = relay.transform.InferType()(mutated_mod)
var_7668 = relay.var("var_7668", dtype = "float32", shape = (15, 4, 15))#candidate|7668|(15, 4, 15)|var|float32
uop_7669 = relay.sigmoid(var_7668.astype('float32')) # shape=(15, 4, 15)
func_4590_call = mod.get_global_var('func_4590')
func_4593_call = mutated_mod.get_global_var('func_4593')
const_7679 = relay.const([-6.739463,1.003907,5.796344,-9.944006,0.713625,8.606806,0.686091,6.286999,-3.481108,-6.346098,-8.861859,-7.283159,-4.929263,-3.248479,8.923438,-1.594630,4.763798,5.983041,0.271229,6.789297,0.986754,7.356868,7.144919,-7.099533,-1.692147,-2.582533,-0.502098,-1.470766,9.552594,-7.934710,-0.385837,5.633298,-4.876541,-8.137633,-6.272930,8.744577,-7.880936,-4.111963,0.979570,8.112896,6.569486,-2.047658,-0.483893,-7.424700,-8.557518,-0.908346,6.394355,-9.794609,-4.130418,4.420308,-1.019004,-4.924729,8.736908,3.583318,2.866665,-8.518900,-4.011524,-0.408720,9.066354,-7.169858,2.305536,-9.054480,-4.899018,6.290402,5.219746,3.069029,5.624654,-1.013353,8.648389,5.842821,-2.585756,9.388859,0.636032,5.088190,0.426291,-1.131020,-6.847212,1.117634,7.879519,5.887458,-1.854851,-9.312530,2.177623,-1.826435,-2.668201,9.820134,-3.194859,6.092721,7.936522,-8.191989,1.769880,5.282741,-1.584625,-2.188010,3.506532,-1.572390,5.996517,2.322900,-1.187795,0.233878,-7.586067,5.900287,2.620552,-2.996331,-5.562115,8.619673,9.069183,5.198884,-5.604865,3.777594,-3.193669,4.332573,-3.165234,-7.146434,-2.742022,-9.999616,-7.657719,9.236124,-4.923218,-8.699236,0.991273,-5.343026,5.201414,-7.569549,-5.021541,3.221192,7.245395,-5.692055,-0.238707,-4.270278,-5.060116,3.805255,0.860412,-1.516856,-5.225022,-9.313162,9.533212,-2.200222,-6.320860,-6.530194,-2.637522,2.522178,-4.050399,8.880450,-4.481515,4.859153,-2.213308,-3.967441,-6.887033,-7.315806,9.352476,-2.701963,3.842182,6.936318,8.013579,-0.948741,-4.714135,1.506076,5.935446,0.555948,3.364062,-9.473972,4.318231,6.293791,-8.295209,-8.333871,-4.197359,-2.450699,3.655478,1.378444,3.259666,-5.573870,-7.611892,-6.703325,-5.178459,9.239611,-1.745973,-1.179806,-7.985385,8.218035,5.859245,1.039127,-7.212539,2.565769,5.190391,-0.194038,8.904054,-6.309202,2.681304,4.214756,-4.373470,8.254698,7.988633,-6.057439,2.207726,-8.221862,8.588231,-6.688053,6.582195,5.572086,-6.718310,1.289364,-9.607266,1.413207,-8.858222,-6.207948,-4.576982,0.553509,2.963506,8.264837,-5.618404,1.432649,8.429557,3.243290,6.840854,-0.758449,3.660113,0.167353,8.960044,2.382001,-8.799271,3.632240,6.412675,-9.120656,-0.576659,3.639113,9.335403,9.206288,-5.980973,1.681005,8.608823,1.116611,4.499816,3.325580,-7.808328,-0.707307,-9.634103,0.308584,-6.429641,-4.049976,8.394501,8.082628,-0.301874,-8.667351,-0.167208,-6.359731,8.056605,-5.713830,2.822550,4.481807,-1.271425,-1.072504,4.421708,-3.832723,9.864859,3.031839,-9.838507,1.799013,9.553601,1.723119,6.882683,8.814230,4.184879,8.947052,8.784593,-3.591940,-5.428374,-9.287474,-6.553810,7.564081,7.770405,3.282530,-1.791062,-5.436436,-4.009688,1.218149,-8.902518,-8.260676,-7.853589,-1.294091,-4.762453,2.627982,4.254257,6.953471,-9.383750,6.127958,4.457929,9.795567,3.522776,6.835279,4.676229,8.770009,5.015241,0.319550,7.623985,-0.351739,5.777914,-2.152554,9.855084,-3.540696,-6.132708,-7.756096,-4.339777,5.929528,9.235560,5.879784,8.475860,-9.004999,-9.382761,-7.766934,0.938285,-0.738831,-6.277576,8.508740,-4.987686,1.001969,4.915688,-7.277731,3.878705,5.881453,-6.770741,3.824944,3.949648,-2.355450,-3.315495,8.648660,4.822821,-8.013629,-8.672926,-0.892925,0.639190,-1.422500,5.624788,4.093835,-9.720027,9.982682,1.964491,3.273456,6.008568,0.230444,-2.830389,2.759340,-5.473551,-0.748142,-4.895629,1.198152,-7.356070,-2.958921,-7.647004,4.774311,4.105528,3.874753,2.803554,3.627207,-6.676902,-0.298332,-4.828786,5.106397,0.578325,5.387026,-1.076679,-8.963046,-3.177262,-1.295700,-6.572101,-5.465713,-9.193234,-2.068924,-7.794354,-2.787988,-1.197090,9.835856,0.242041,5.939670,-6.131794,-4.966334,6.653826,-4.543385,0.476242,-9.807022,-5.665932,4.845262,-7.985519,-0.451313,-5.905418,-1.878437,-7.821692,6.611918,-6.667745,-5.076368,4.782665,5.111887,2.797765,3.243283,-7.844489,-8.930277,-2.764758,0.766804,-1.109797,6.720917,9.173236,0.394236,1.486063,7.688079,-5.865540,0.017956,-5.864109,-5.657533,-0.313055,-2.625385,-3.777578,1.457799,8.546495,0.754833,-9.930924,-4.733842,-2.514704,6.459735,-6.575440,0.167081,9.912968,-0.366850,7.419259,6.289439,-4.150982,-9.929555,7.863346,6.204057,5.339790,-4.125979,6.803682,5.033970,-8.007679,-3.276183,4.372044,1.384202,-5.148103,-4.992429,0.319424,8.445041,7.118872,-6.875067,2.148552,0.802346,-3.050449,-3.277856,-0.928122,-6.976458,8.226224,1.303640,-5.026430,7.757029,6.743181,9.261595,6.456217,-5.902272,-0.034122,7.967157,8.220441,-2.696206,-0.311799,7.072803,-9.523958,-2.004172,-2.085377,-1.725052,-7.805752,4.016786,8.337097,-5.026240,7.316991,-2.409680,0.652679,2.353341,-8.153144,8.355928,0.093847,-6.975900,-7.452804,8.639769,-4.044494,4.373572,8.244276,-9.734597,9.936339,4.318273,-9.214926,3.268483,-3.038357,-7.010074,-4.813275,4.192903,-4.073610,-0.319762,-5.016218,-8.960517,5.020006,-1.562660,-3.381509,-0.653798,7.994650,5.698901,1.487161,-8.890652,6.838278,-8.525385,-1.555743,2.126580,9.045597,7.809583,0.872377,4.652602,8.081191,8.920869,-2.946078,-4.305678,-1.613374,5.131016,-3.701910,1.983660,-8.492574,5.484421,0.068076,-3.133703,-4.006878,1.322823,4.229676,2.242255,-9.178302,-7.928887,-5.234659,-3.694624,-7.134332,-1.398514,2.891614,6.550255,6.082920,-1.587302,6.350604,-2.114737,-0.705360,-0.741298,-0.498928,8.178895,-3.367416,6.616193], dtype = "float64")#candidate|7679|(546,)|const|float64
call_7678 = relay.TupleGetItem(func_4590_call(relay.reshape(const_7679.astype('float64'), [3, 13, 14]), relay.reshape(const_7679.astype('float64'), [3, 13, 14]), ), 1)
call_7680 = relay.TupleGetItem(func_4593_call(relay.reshape(const_7679.astype('float64'), [3, 13, 14]), relay.reshape(const_7679.astype('float64'), [3, 13, 14]), ), 1)
func_4455_call = mod.get_global_var('func_4455')
func_4458_call = mutated_mod.get_global_var('func_4458')
var_7695 = relay.var("var_7695", dtype = "float64", shape = (441,))#candidate|7695|(441,)|var|float64
const_7696 = relay.const([10,6,-7,3,-6,-6,-1,1,4,7,-9,-10,8,8,-7,-10,-6,-10,3,-7,-6,-10,-1,-10,-8,2,8,10,6,-6,4,1,2,7,7,-8,4,-1,9,-9,-10,6,3,-6,5,-1,7,3,3,7,-1,-4,3,1,9,-8,-10,-7,-2,1,4,-8,-4,-2,-10,-7,-6,-7,8,-10,5,6,-4,-8,7,-4,1,-5,3,-10,-8,4,-1,4,7,6,4,9,6,-4,7,5,-2,-3,-8,7,-3,2,4,-3,10,4,3,1,3,4,3,-9,2,9,-6,-7,-1,-1,5,-5,7,-10,-5,2,6,9,9,-4,10,10,2,-5,4,-8,-3,-7,-4,-1,-8,8,-10,3,5,-10,-3,-2,-6,9,-6,1,2,8,-5,-7,-1,-8,2,-2,-1,3,-1,-10,-6,8,-8,9,-7,5,5,8,7,-9,-7,7,1,3,-8,10,-2,6,-7,7,2,-9,-6,-3,1,5,2,8,-3,-7,2,5,-4,-3,5,-7,3,-6,-1,3,2,1,-9,7,-8,3,10,3,10,-6,-5,4,9,7,10,3,10,-3,-3,-1,3,-10,-3,-8,8,-7,-7,2,-10,7,3,-1,-5,1,7,10,7,4,4,2,5,6,-10,-1,7,2,9,-2,10,-4,-9,-6,4,-6,-1,2,-8,10,-7,10,-9,6,6,4,-3,-2,3,-9,-9,7,5,-7,-6,-3,-10,-8,5,-8,-4,-8,3,9,9,4,5,9,-9,-7,3,-9,-4,1,3,-10,-4,-3,-10,4,2,-9,-1,1,3,6,-9,-1,6,-10,-3,-7,3,-7,-4,-4,1,7,2,-4,7,2,-7,-7,-5,3,1,9,9,7,-7,2,-4,9,-9,2,-3,2,-7,4,-7,-4,-6,-4,-8,-2,4,2,5,6,6,-9,-2,2,9,-3,-4,7,-9,9,-4,2,-5,5,-10,3,9,-7,-5,10,7,-2,-2,5,2,1,-2,-3,9,-8,-1,7,7,8,-4,-5,4,4,-6,7,10,5,1,3,-10,-9,4,-10,1,-2,-8,6,2,-8,-8,8,5,3,1,-7,7,-1,-3,1,-9,-9,-9,1,-6,10,5,3,1,-9,-2,-3,-4,1,3,3,6,-1,8,6,10,-7,4,-7,6,7,-2,-6,-3,8,-8,10,-5,-6,-10,7,8,4,-4,7,7,-5,-2,-7,-7,-3,-8,-8,5,-7,8,-4,8,10,-9,9,1,-6,6,-10,-8,9,-9,10,2,-8,1,-9,-9,8,-6,-5,2,1,-5,-2,-7,1,-7,7,-10,2,-6,10,-7,-5,-7,9,-2,-9,1,-9,4,4,3,-1,-8,-4,-3,2,-3,3,9,-1,-4,5,-4,6,6,-9,1,2,2,-6,2,-5,4,7,3,2,10,5,-6,2,-10,3,-1,-3,-2,-8,-6,9,2,-3,6,2,-5,-10,7,4,-10,-1,1,4,4,8,-6,9,-5,-4,-8,-6,3,-7,-5,10,-8,-6,8,2,3,-3,1,6,-4,8,7,8,-10,-2,6,2,-6,10,-8,7,4,-8,-2,10,5,10,2,3,1,-8,-1,10,7,-3,-2,-8,-10,3,3,8,3,-1,5,8,-1,-4,5,10,-2,8,-3,4,3,7,-1,5,2,1,-9,-3,5,3,-9,-3,7,-7,-8,-6,-8,-10,-8,5,10,-7,-6,1,8,-1,-8,8,-6,8,7,6,8,6,-10,6,-8,-10,3,-6,-7,8,9,-3,-2,-8,4,6,-1,-8,7,10,-8,2,6,-4,2,10,-9,-6,-3,-2,10,1,-4,2,-4,7,-5,-5,6,3,1,-1,8,1,-4,10,-3,-7,1,-8,-4,-9,5,-1,3,-9,9,5,7,-9,-1,7,4,-10,-5,-10,-3,-9,2,9,4,-2,-10,-10,-4,-10,-2,8,-4], dtype = "int8")#candidate|7696|(729,)|const|int8
call_7694 = relay.TupleGetItem(func_4455_call(relay.reshape(var_7695.astype('float64'), [7, 7, 9]), relay.reshape(const_7696.astype('int8'), [729,]), ), 1)
call_7697 = relay.TupleGetItem(func_4458_call(relay.reshape(var_7695.astype('float64'), [7, 7, 9]), relay.reshape(const_7696.astype('int8'), [729,]), ), 1)
uop_7709 = relay.erf(var_7668.astype('float64')) # shape=(15, 4, 15)
func_6251_call = mod.get_global_var('func_6251')
func_6253_call = mutated_mod.get_global_var('func_6253')
call_7711 = func_6251_call()
call_7712 = func_6251_call()
output = relay.Tuple([uop_7669,call_7678,const_7679,call_7694,var_7695,const_7696,uop_7709,call_7711,])
output2 = relay.Tuple([uop_7669,call_7680,const_7679,call_7697,var_7695,const_7696,uop_7709,call_7712,])
func_7718 = relay.Function([var_7668,var_7695,], output)
mod['func_7718'] = func_7718
mod = relay.transform.InferType()(mod)
mutated_mod['func_7718'] = func_7718
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7718_call = mutated_mod.get_global_var('func_7718')
var_7720 = relay.var("var_7720", dtype = "float32", shape = (15, 4, 15))#candidate|7720|(15, 4, 15)|var|float32
var_7721 = relay.var("var_7721", dtype = "float64", shape = (441,))#candidate|7721|(441,)|var|float64
call_7719 = func_7718_call(var_7720,var_7721,)
output = call_7719
func_7722 = relay.Function([var_7720,var_7721,], output)
mutated_mod['func_7722'] = func_7722
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4770_call = mod.get_global_var('func_4770')
func_4772_call = mutated_mod.get_global_var('func_4772')
call_7734 = relay.TupleGetItem(func_4770_call(), 2)
call_7735 = relay.TupleGetItem(func_4772_call(), 2)
output = call_7734
output2 = call_7735
func_7737 = relay.Function([], output)
mod['func_7737'] = func_7737
mod = relay.transform.InferType()(mod)
output = func_7737()
func_7738 = relay.Function([], output)
mutated_mod['func_7738'] = func_7738
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7259_call = mod.get_global_var('func_7259')
func_7260_call = mutated_mod.get_global_var('func_7260')
call_7867 = relay.TupleGetItem(func_7259_call(), 0)
call_7868 = relay.TupleGetItem(func_7260_call(), 0)
output = call_7867
output2 = call_7868
func_7894 = relay.Function([], output)
mod['func_7894'] = func_7894
mod = relay.transform.InferType()(mod)
output = func_7894()
func_7895 = relay.Function([], output)
mutated_mod['func_7895'] = func_7895
mutated_mod = relay.transform.InferType()(mutated_mod)
var_7926 = relay.var("var_7926", dtype = "float64", shape = (11, 16, 9))#candidate|7926|(11, 16, 9)|var|float64
uop_7927 = relay.erf(var_7926.astype('float64')) # shape=(11, 16, 9)
func_7718_call = mod.get_global_var('func_7718')
func_7722_call = mutated_mod.get_global_var('func_7722')
const_7936 = relay.const([1.892533,5.214662,2.988906,6.979961,-8.694894,3.314228,-7.826748,5.810901,-8.524849,9.596012,-8.554047,-2.944042,7.333002,-8.537239,4.311257,-8.420481,6.664554,-7.305571,-4.974547,-4.708877,-1.799433,6.290659,-0.841319,-2.281313,4.156797,-8.117504,-2.699223,-3.941157,6.580297,-2.037960,-4.731639,7.195062,-1.750319,3.413820,3.469411,8.179762,5.638380,2.179853,4.099601,-9.728897,7.565782,-3.481614,-9.547725,3.277142,-6.133237,-6.069881,4.420485,-8.346533,6.470438,-4.644710,8.446073,-5.716439,-1.976915,-9.259484,1.913700,0.609917,-3.004498,-4.377001,-2.011902,-4.171127,0.045712,-4.470885,3.031991,-1.831828,-8.244765,0.840304,-4.961508,-5.583255,9.999920,-6.470329,6.594001,-3.057684,2.797228,-1.452692,-2.760566,-9.584224,0.557062,-4.963735,0.363600,0.710012,1.841462,6.550911,-9.693342,4.361942,7.389647,-3.933013,1.306095,-5.247066,3.892849,-9.922314,6.881019,7.355612,1.412091,2.050686,-0.454562,-8.694194,-1.371373,0.975048,8.024094,-3.389398,-4.283647,-2.721756,5.174205,8.435598,0.298148,4.845177,-6.869500,-5.932987,-4.605424,4.729042,-7.319128,6.094467,0.033775,-2.276878,5.977228,1.392342,6.367280,4.332261,-1.678532,-3.293246,-9.350441,9.270818,-0.528182,9.681536,4.793987,-5.922188,-9.199886,-1.564665,-4.582986,7.424236,-2.940490,-7.083846,-1.317032,-9.479620,-3.702252,-6.378975,-2.672438,0.157476,0.245277,-8.389981,8.556118,8.800166,-5.899828,-3.595776,-3.926221,-9.486606,-7.675021,-4.536329,-8.145081,-0.776307,5.360509,0.282755,-5.759437,6.631679,6.415337,-3.253466,-3.913936,4.338529,-1.706740,9.920274,0.608353,6.214468,-4.198911,-3.683404,-8.290370,3.947265,2.006456,-7.192763,-8.318976,-3.921005,3.947703,-8.069287,-9.261171,-7.665061,3.211640,7.038122,8.528455,7.432588,-5.531201,-2.082192,-0.230030,4.558870,-2.966719,4.385340,7.947347,7.862389,-3.885224,-3.628615,7.186944,4.586655,-3.598836,4.670511,3.700318,-9.694716,-5.346683,1.420460,-2.370755,-5.489861,-3.489172,0.657245,5.438726,-7.081847,0.444457,-4.030168,6.458179,9.412737,-6.362351,1.261349,3.024947,-5.645607,-4.834635,-8.804671,-5.065811,2.971649,-1.255861,-5.740656,1.099286,-6.211131,7.729301,6.174770,-7.017980,-6.944294,1.165639,7.733082,0.790147,-1.806191,9.078353,7.231280,-4.579185,-1.155814,7.007845,-5.824207,5.790395,4.900963,-4.093676,0.280765,-6.805544,4.485981,7.194122,-9.440605,-0.396035,-5.128858,-0.179485,8.554509,-8.462235,8.586567,-8.662940,2.737023,-9.763983,-4.636271,0.697767,-9.428050,-7.060613,-0.163667,-1.977186,-2.499119,8.073368,-6.430937,-6.674311,-5.957022,4.511585,9.590622,-2.083525,4.851750,8.547810,0.947457,1.856649,-5.753112,2.066911,0.997108,-1.419049,1.299434,-9.031234,5.446505,0.370422,2.718951,9.685319,2.885529,4.563792,0.442073,-2.560392,-3.296359,-5.817467,6.791313,3.196502,5.027359,4.310498,9.423315,-8.224472,-3.793626,8.120534,-1.758619,-1.926528,-3.805683,-8.035037,5.515897,-7.256572,-5.886816,-6.107663,-5.423919,-5.278917,2.411127,-9.991685,2.128462,6.029775,7.777881,3.731981,-8.635697,7.998615,-5.712841,-2.166202,-3.609111,0.835062,1.942072,3.987374,6.106209,9.226466,8.411546,6.658158,-4.552492,-8.888928,5.130625,4.499918,-3.989811,-5.792033,-9.788500,-0.190165,5.169965,4.473409,6.718098,9.482962,-1.649944,-1.186924,-8.613240,-9.797488,-0.428639,-9.173298,-3.307346,2.882269,-9.742386,7.355332,-2.882733,-3.026958,8.982588,-7.164866,-3.173269,4.731691,-4.792993,-2.965970,-6.802989,7.378887,5.872422,-9.137618,4.074337,-3.035715,-1.599691,-0.136744,4.329760,-3.313504,-0.486017,-9.834419,-0.989562,0.637743,1.816339,-9.021579,-2.942512,-8.852871,5.091756,-3.665394,9.096394,9.166165,-0.755879,-7.941236,-6.699112,3.256519,-4.853465,6.804795,6.937859,8.358225,-0.492442,-1.603575,-2.450800,-6.770160,-9.928629,-9.223431,-1.609541,9.198680,-5.837737,-9.967459,-2.379880,2.604836,1.238440,-3.679783,-9.393111,7.387029,9.129491,9.627220,1.709755,4.992017,-3.385117,1.982927,6.311913,-3.563866,-3.749439,-8.610618,-8.748794,3.043164,-6.592049,-2.839675,-9.659032,7.517328,-5.253269,8.987291,3.407029,3.566707,-3.952873,-9.415968,6.015275,7.534191,6.563294,-1.826640,-0.652051,-9.978778,3.803065,8.563145,-3.848471,5.333666,7.338090,-3.525626,-0.928943,6.438033,-2.984425,-4.095273,-1.743987,-1.179882,-9.404275,4.726821,-8.174098,4.982087,-6.735289,3.426833,-5.228355,5.614473,2.428941,-6.236247,2.622048,6.476265,2.785302,9.621790,9.241300,5.851657,6.476497,-2.095034,-9.884456,0.811417,-4.813254,-8.426940,-8.250202,6.060412,8.536129,3.831331,7.259648,9.212706,-0.643097,-4.124963,-5.282318,6.339444,-7.819653,4.070931,-2.340903,7.764557,-9.109466,9.745129,-9.553799,-6.905557,-6.297162,-1.592764,-4.400824,-2.479673,7.166475,6.755781,-7.942026,5.907463,-9.033061,-0.604726,3.495436,-7.023250,4.767315,-9.758188,-5.415539,1.735240,-6.089668,-4.324214,1.212672,-1.953505,9.597889,-0.862007,-6.388386,-9.739850,5.604739,4.644781,-5.981287,-5.883285,4.272010,6.586270,2.010067,-4.992810,7.945454,8.937265,4.676140,0.146055,7.601236,3.100974,6.060095,-6.744595,-4.931905,3.764810,3.255010,-9.639471,4.922645,-5.038299,-1.711061,-1.782100,4.293319,1.257231,8.347819,8.559801,-8.853549,0.791909,-8.992784,-6.392289,6.724385,-9.380523,4.523057,-3.913433,-7.249688,-4.923009,2.916127,-4.988640,-9.056293,-2.621670,-2.229392,-3.407966,-2.145890,-2.480875,1.833269,9.670186,6.960995,-7.226811,-1.075869,6.080860,6.182214,3.281301,-1.030289,-2.774725,1.070770,3.868267,-2.467524,4.747119,6.158378,-3.332292,-2.675276,0.104953,0.914578,6.759588,-3.802032,-6.898857,-8.948691,2.645720,-4.233421,0.515842,-1.548641,6.064596,-6.211322,9.889941,1.462701,-1.607246,1.279431,0.741001,6.493705,-4.763632,4.792404,-0.954317,-9.531593,-2.998705,2.949298,7.968791,-2.447184,-1.333902,7.515431,-1.070754,1.949080,-7.928765,4.152126,3.231582,-7.725828,-9.072631,0.683084,7.455998,-0.279299,-4.292025,8.860753,9.513680,-1.413043,-6.152287,6.793272,-4.050149,-2.091133,0.634916,-8.071918,-7.992223,-9.853616,-6.362143,5.035309,-0.787215,-1.499280,7.496685,-5.913792,1.954818,8.715169,-3.846461,6.162278,6.285419,0.860118,6.932646,-3.077233,8.868982,-3.267972,-8.075911,5.116607,-8.957366,0.082877,-6.123323,-5.342533,-5.325859,3.910785,8.994524,-2.061548,-9.376807,6.006971,8.646412,-3.078298,1.864226,8.962159,-4.752281,-6.941195,-6.410335,1.305735,6.645094,-0.541547,7.248826,-5.416814,-7.715995,9.852533,-5.614707,-5.487959,-0.897873,-4.733893,-2.727919,8.570465,5.833530,3.022740,0.509055,-6.868298,-4.624124,7.705889,-2.355542,-7.751911,-6.663229,-7.466063,9.186269,-7.106766,5.267920,2.234202,-3.200716,-2.515601,-8.055999,8.926960,-1.627819,-3.557757,8.658442,-6.831727,1.440757,-7.425528,5.155030,-3.562338,-4.909561,-1.463195,-6.847399,9.362282,8.848314,-3.149959,1.265426,0.393381,2.892032,-2.908579,0.518321,-1.602939,-3.978296,-7.542262,1.578440,-9.017081,-5.368553,9.889153,-8.591785,2.988397,7.279282,9.910640,-6.145069,-8.739934,-8.428660,-5.991254,-6.277646,-7.141796,6.699333,-5.059340,3.059553,-8.711136,1.499555,-1.801853,-2.507185,8.973392,9.539865,-3.926256,-0.705591,3.344699,-1.372607,-6.980849,5.248427,9.651978,-0.675136,8.209598,-1.707003,8.129062,-5.363133,7.744088,5.423585,-8.212421,9.492549,-0.707398,2.323626,-3.450040,7.765586,-2.823886,8.724715,-2.407820,3.719916,3.360012,2.263687,-6.117336,4.927657,6.567273,-9.728447,4.679715,-8.630769,6.731918,-0.712583,0.431461,6.508610,-6.532752,1.724402,-2.315028,-1.855087,-4.721726,-0.379580,1.991980,-7.097234,-7.021932,-1.572078,-1.677615,-9.044037,3.194427,9.146571,-5.387796,-2.861762,9.886271,4.333618,4.948819,-9.924931,-9.755532,-9.611772,-3.184033,-4.253124,7.844443,-4.027460,3.177329,-2.638191,7.664586,-2.378349,-6.944974,-7.446234,-9.521563,5.912544,-4.083927,-3.612647,-7.058929,-1.054935,6.064369,2.417180,-4.903124,1.639456,-0.931254,-6.695338,2.818043,-4.334291,6.676492,-9.731125,-6.409146,7.919990,6.881470,2.092513,-0.617760,-3.861236,-8.432741,3.867617,7.237990,5.706939,-5.290195,-8.139209,-2.700074,3.069533,1.887743,8.636825,4.781294,-4.285210,-3.719352,-6.439230,-6.842911,4.339359,-9.888180,8.764976,7.308984,-6.454195,-5.485794,9.563588,6.604856,-1.212483,-6.058370,-5.021956,7.132844,-9.489375,3.636099,-7.513564,0.525450,-6.206818,-6.634679,-2.395761,-7.094248,9.802722,2.733975,3.996812,1.451356,5.104169,-5.085732,7.599499,0.922371,4.311520,-2.349673,-3.555029,2.768716,-1.803791,-9.407864,1.078199,8.376170,7.824562,-8.856589,2.340641,-6.074287,7.144959,-4.893574,-9.270667,5.135164,-7.658446,-6.273654,9.453335,-9.061441,2.852034,-3.427156,8.804291,-6.113405,-9.877703,-1.197208,-9.288761,8.202358,3.492907,5.436918,3.693023,6.020793,-2.269860,-8.752732,-3.382344,3.740294,9.586436,-3.478163,3.793480,1.068108,7.235980,8.884090,-9.670784,-8.812307], dtype = "float32")#candidate|7936|(900,)|const|float32
const_7937 = relay.const([[-3.779487,8.119201,-1.048042,3.542059,4.222715,-7.487706,-7.462905,-3.970702,2.422867,0.938717,-8.457396,7.552826,1.644046,-4.664484,2.008872,1.120695,0.929505,-1.068427,-3.998149,-3.359386,-0.005388,-2.058085,-3.134161,-7.917360,-4.069721,5.482646,-6.732492,7.403748,7.528572,0.821038,-4.234816,-2.305303,5.623514,-6.643678,0.187444,6.098950,5.565599,-3.313268,7.208562,3.454328,-9.394622,-0.414687,-5.963750,-2.847869,3.907219,-5.672773,-3.950761,8.387318,-4.806883,-5.266404,-6.387052,3.825426,4.476454,-2.773803,3.346950,-0.780246,-4.296758,-9.863752,9.140829,-2.656933,-7.619550,-5.446416,7.473930,-8.136458,7.476434,9.962186,-1.418594,3.957043,-5.601131,-7.625011,-0.825378,4.269063,2.272711,7.969849,-4.686284,8.562231,-9.047969,-0.117086,-8.324189,8.518827,2.873621,7.953384,-1.470720,5.172343,-1.171519,-2.543683,-3.372774,2.341629,4.868704,-0.980028,-0.209817,-4.081375,-5.672140,2.298285,8.347407,-8.397665,8.571321,-5.056750,1.762587,6.367817,-3.332752,-9.476715,7.225719,-7.426353,-9.102396,9.135240,6.913031,4.296448,-7.413582,-1.223843,-7.646350,-2.090210,-2.429862,5.843349,-9.801722,-8.857071,-2.845409,7.758567,-6.594278,-8.726927,4.494437,-5.681202,0.597883,-9.171764,-4.730817,4.178677,4.520557,-0.166996,-6.586297,7.633207,-6.565893,4.379207,-1.939133,6.065944,9.004193,7.269405,8.466873,5.083489,-0.900560,-1.090149,-6.881407,9.641872,8.555963,5.213937,0.007499,-7.550935,-5.650251,-8.350651,-4.490614,-0.994894,0.998168,-7.308482,8.231081,4.937138,5.230884,-9.552963,7.773064,-6.439381,4.424476,-0.251683,-0.831747,8.183598,7.371269,-6.041236,1.458331,8.541813,-6.507971,2.644396,4.723072,4.226894,-2.636923,-2.462205,1.130373,-2.151852,-8.494113,7.450826,4.975672,-4.077348,-8.754894,-1.094511,8.134089,-8.185782,-4.029706,4.322185,-1.010955,-5.138221,-9.546100,-2.423019,5.348533,-9.571476,-8.958647,3.399578,-2.676167,-7.691464,-2.685976,-9.808721,-6.560547,1.965893,-2.386392,-2.448706,-1.199208,5.842594,6.414348,-1.333437,-9.141992,-9.327432,1.154226,-6.983343,5.145116,-1.861013,-1.476258,8.706070,-3.720880,4.997315,2.684297,1.343208,-4.308028,-0.522251,5.811687,9.803367,-4.251792,-8.532331,9.732521,1.226931,5.800893,-4.626543,0.846440,-4.017660,-9.647170,-4.769602,9.979725,4.557743,6.214680,6.673252,7.373253,-0.143185,-8.464949,9.894940,-8.808221,3.534589,-0.453179,0.973188,1.715210,6.964486,2.702651,-6.413039,-6.976593,-9.879194,0.754828,-6.133032,-5.405727,-0.278631,6.286979,-2.708424,1.401322,4.801843,1.533859,6.154009,-5.205997,6.694067,-2.653512,-3.890970,8.354483,-9.893794,-1.029538,5.707288,-7.140578,0.075055,1.140126,1.287940,0.503512,4.687824,2.203281,-4.714017,5.912993,7.632791,4.312199,-5.868469,-4.664647,2.171687,0.199172,-8.044153,2.515611,5.123125,0.476954,0.385733,7.413160,-4.228822,4.486902,-5.118298,-2.868902,-0.614040,9.893115,-0.487724,-4.265005,-1.291060,9.109239,9.236484,9.935291,0.840853,7.840743,3.568505,5.515947,9.288531,-6.405139,-0.671145,0.490832,-0.244420,0.194115,-4.996673,8.476296,-4.968291,0.228616,7.187744,-8.310827,-4.645020,3.840059,2.778476,7.573346,1.435333,-0.501728,-1.866997,-9.667175,0.146209,2.143660,7.416158,-6.363998,9.512226,0.588155,-6.740123,-8.377438,2.161895,1.209143,-3.336821,1.065013,-7.370396,9.032324,4.922062,-9.924111,-5.486229,3.442178,-3.080867,8.480515,8.490448,-5.850731,-4.958393,-4.436228,-4.078755,-9.349836,-1.837788,-6.542459,-1.160963,8.532784,7.690271,-8.073763,-9.185183,-0.331199,2.066406,-1.197858,-4.226476,5.215291,3.655136,9.701955,3.407204,0.472069,7.303073,-7.340024,7.510518,-8.510948,4.748835,-9.993876,-7.448183,-3.024921,-4.665074,-1.139276,0.791588,-3.526292,-9.987842,-0.256556,4.319445,9.140719,3.279906,-3.361003,-5.702710,-8.887940,9.916260,8.148285,4.998003,-2.721984,-4.292113,-9.317704,-1.181522,2.761403,-5.633168,-0.394465,2.411367,7.493813,9.163569,4.818737,-3.440606,-1.164741,7.571445,3.149796,5.658329,-1.324817,-5.697430,-2.633476,1.359676,6.271272,7.673579,-1.705087,-1.714042,-9.468131,2.046440,0.958454,5.809423,-3.714000,6.746959,9.052210,4.955500,-4.326170,-9.713500,-3.760945,-9.079706,-9.367685,5.302171,-1.817706,-1.141058,4.408439,7.838048,7.304043,-7.739909,9.765609,6.583702,8.060626,-2.925926,-5.569851,-8.675409,6.866407,9.803835,-7.495496]], dtype = "float64")#candidate|7937|(1, 441)|const|float64
call_7935 = relay.TupleGetItem(func_7718_call(relay.reshape(const_7936.astype('float32'), [15, 4, 15]), relay.reshape(const_7937.astype('float64'), [441,]), ), 1)
call_7938 = relay.TupleGetItem(func_7722_call(relay.reshape(const_7936.astype('float32'), [15, 4, 15]), relay.reshape(const_7937.astype('float64'), [441,]), ), 1)
output = relay.Tuple([uop_7927,call_7935,const_7936,const_7937,])
output2 = relay.Tuple([uop_7927,call_7938,const_7936,const_7937,])
func_7940 = relay.Function([var_7926,], output)
mod['func_7940'] = func_7940
mod = relay.transform.InferType()(mod)
mutated_mod['func_7940'] = func_7940
mutated_mod = relay.transform.InferType()(mutated_mod)
var_7941 = relay.var("var_7941", dtype = "float64", shape = (11, 16, 9))#candidate|7941|(11, 16, 9)|var|float64
func_7940_call = mutated_mod.get_global_var('func_7940')
call_7942 = func_7940_call(var_7941)
output = call_7942
func_7943 = relay.Function([var_7941], output)
mutated_mod['func_7943'] = func_7943
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5968_call = mod.get_global_var('func_5968')
func_5970_call = mutated_mod.get_global_var('func_5970')
call_7969 = func_5968_call()
call_7970 = func_5968_call()
func_6088_call = mod.get_global_var('func_6088')
func_6090_call = mutated_mod.get_global_var('func_6090')
call_7972 = func_6088_call()
call_7973 = func_6088_call()
output = relay.Tuple([call_7969,call_7972,])
output2 = relay.Tuple([call_7970,call_7973,])
func_7976 = relay.Function([], output)
mod['func_7976'] = func_7976
mod = relay.transform.InferType()(mod)
output = func_7976()
func_7977 = relay.Function([], output)
mutated_mod['func_7977'] = func_7977
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7737_call = mod.get_global_var('func_7737')
func_7738_call = mutated_mod.get_global_var('func_7738')
call_8011 = func_7737_call()
call_8012 = func_7737_call()
var_8015 = relay.var("var_8015", dtype = "float64", shape = (660,))#candidate|8015|(660,)|var|float64
bop_8016 = relay.logical_and(call_8011.astype('bool'), relay.reshape(var_8015.astype('bool'), relay.shape_of(call_8011))) # shape=(660,)
bop_8019 = relay.logical_and(call_8012.astype('bool'), relay.reshape(var_8015.astype('bool'), relay.shape_of(call_8012))) # shape=(660,)
uop_8028 = relay.sinh(call_8011.astype('float32')) # shape=(660,)
uop_8030 = relay.sinh(call_8012.astype('float32')) # shape=(660,)
output = relay.Tuple([bop_8016,uop_8028,])
output2 = relay.Tuple([bop_8019,uop_8030,])
func_8031 = relay.Function([var_8015,], output)
mod['func_8031'] = func_8031
mod = relay.transform.InferType()(mod)
var_8032 = relay.var("var_8032", dtype = "float64", shape = (660,))#candidate|8032|(660,)|var|float64
output = func_8031(var_8032)
func_8033 = relay.Function([var_8032], output)
mutated_mod['func_8033'] = func_8033
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7480_call = mod.get_global_var('func_7480')
func_7481_call = mutated_mod.get_global_var('func_7481')
call_8045 = relay.TupleGetItem(func_7480_call(), 2)
call_8046 = relay.TupleGetItem(func_7481_call(), 2)
output = call_8045
output2 = call_8046
func_8058 = relay.Function([], output)
mod['func_8058'] = func_8058
mod = relay.transform.InferType()(mod)
output = func_8058()
func_8059 = relay.Function([], output)
mutated_mod['func_8059'] = func_8059
mutated_mod = relay.transform.InferType()(mutated_mod)
var_8072 = relay.var("var_8072", dtype = "uint8", shape = (10, 9, 6))#candidate|8072|(10, 9, 6)|var|uint8
const_8073 = relay.const([[[-1,-10,2,-3,-5,7],[2,-6,2,-7,4,-8],[-1,3,-9,-1,-1,-1],[-1,2,-7,-9,1,-5],[9,-5,6,-9,10,-8],[4,-1,-2,-8,7,10],[10,-3,5,5,-10,6],[2,4,7,2,1,4],[8,-6,-3,8,-6,-4]],[[7,-2,10,6,3,-5],[-1,-10,4,8,8,-4],[-10,6,-1,6,10,8],[6,-3,3,-7,-6,-9],[-5,-2,-10,-5,9,8],[-2,2,2,3,4,-7],[3,-3,5,1,8,-2],[-10,-1,-10,2,5,3],[4,7,1,8,10,-5]],[[5,2,9,-2,-2,-9],[3,-4,5,-6,-3,-10],[-7,-8,-10,-9,-9,8],[6,9,6,2,-9,-8],[4,-5,6,-3,4,9],[4,-10,-10,3,10,-8],[9,-6,6,3,9,-9],[3,-10,-3,10,-9,-2],[-9,-7,-8,7,-4,-6]],[[-8,-2,-8,4,3,8],[-4,-7,1,4,-6,-10],[3,-2,2,-7,8,9],[-10,-9,10,-3,8,-7],[-9,10,1,10,9,1],[1,-4,-9,6,8,4],[-3,-2,5,1,-10,9],[-2,5,4,-5,-6,-5],[1,-4,-9,7,-5,4]],[[-8,-6,-7,-10,-6,2],[-3,-5,3,-3,3,5],[6,-5,-1,-3,-5,9],[-5,-3,-8,-1,2,-9],[9,-1,-9,-6,8,-8],[-2,10,-9,1,10,5],[9,-8,-3,3,-8,4],[-1,-4,2,6,2,-3],[-6,3,6,-3,-3,5]],[[-7,7,3,5,-2,2],[1,-9,-5,4,-7,-7],[-4,-10,-1,-10,-3,4],[-8,1,7,5,-3,-3],[-1,2,-1,-2,9,-9],[-9,9,-7,2,-1,7],[-8,8,-1,2,5,9],[-2,10,-5,2,-1,7],[-3,7,-2,1,-9,3]],[[-9,1,10,-9,-2,-4],[9,2,8,-10,-8,4],[-1,2,-6,-1,10,7],[1,2,7,2,-3,-8],[4,1,-4,4,2,7],[-8,2,-10,10,7,5],[-10,-9,-1,-4,-6,-7],[-2,8,-6,3,-3,4],[6,-5,3,-6,-6,6]],[[4,2,-6,10,1,-5],[3,1,-3,-9,6,-9],[-3,4,1,-9,-1,-3],[-2,-9,1,10,-10,-9],[-7,-1,10,3,-6,-3],[8,-10,-2,10,-7,9],[-10,9,-9,4,-5,-10],[-6,10,-2,9,-4,-1],[-3,2,-9,-8,-9,1]],[[-7,5,-10,9,-8,8],[-2,9,7,4,-10,-9],[-7,-8,-6,-6,3,-7],[4,1,-4,9,-7,2],[10,-3,4,9,-3,-10],[7,-1,5,7,3,1],[-8,-3,-3,-10,-3,-3],[-9,10,3,-8,-6,-1],[7,2,6,7,-8,-1]],[[-7,3,-6,-1,-1,4],[3,-4,-2,-1,3,-1],[7,-6,-2,5,10,1],[4,-10,3,9,-9,-6],[-10,-2,-10,-3,-4,-1],[8,6,-4,3,-9,3],[10,-1,-6,1,-1,8],[-7,8,10,10,7,-5],[2,-9,4,6,-3,-4]]], dtype = "uint8")#candidate|8073|(10, 9, 6)|const|uint8
bop_8074 = relay.logical_xor(var_8072.astype('uint8'), relay.reshape(const_8073.astype('uint8'), relay.shape_of(var_8072))) # shape=(10, 9, 6)
output = bop_8074
output2 = bop_8074
func_8077 = relay.Function([var_8072,], output)
mod['func_8077'] = func_8077
mod = relay.transform.InferType()(mod)
var_8078 = relay.var("var_8078", dtype = "uint8", shape = (10, 9, 6))#candidate|8078|(10, 9, 6)|var|uint8
output = func_8077(var_8078)
func_8079 = relay.Function([var_8078], output)
mutated_mod['func_8079'] = func_8079
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4889_call = mod.get_global_var('func_4889')
func_4891_call = mutated_mod.get_global_var('func_4891')
call_8107 = func_4889_call()
call_8108 = func_4889_call()
uop_8110 = relay.atan(call_8107.astype('float64')) # shape=(14, 2, 10)
uop_8112 = relay.atan(call_8108.astype('float64')) # shape=(14, 2, 10)
output = uop_8110
output2 = uop_8112
func_8121 = relay.Function([], output)
mod['func_8121'] = func_8121
mod = relay.transform.InferType()(mod)
mutated_mod['func_8121'] = func_8121
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8121_call = mutated_mod.get_global_var('func_8121')
call_8122 = func_8121_call()
output = call_8122
func_8123 = relay.Function([], output)
mutated_mod['func_8123'] = func_8123
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5968_call = mod.get_global_var('func_5968')
func_5970_call = mutated_mod.get_global_var('func_5970')
call_8133 = func_5968_call()
call_8134 = func_5968_call()
func_5935_call = mod.get_global_var('func_5935')
func_5936_call = mutated_mod.get_global_var('func_5936')
call_8137 = func_5935_call()
call_8138 = func_5935_call()
output = relay.Tuple([call_8133,call_8137,])
output2 = relay.Tuple([call_8134,call_8138,])
func_8139 = relay.Function([], output)
mod['func_8139'] = func_8139
mod = relay.transform.InferType()(mod)
mutated_mod['func_8139'] = func_8139
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8139_call = mutated_mod.get_global_var('func_8139')
call_8140 = func_8139_call()
output = call_8140
func_8141 = relay.Function([], output)
mutated_mod['func_8141'] = func_8141
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6088_call = mod.get_global_var('func_6088')
func_6090_call = mutated_mod.get_global_var('func_6090')
call_8142 = func_6088_call()
call_8143 = func_6088_call()
func_5968_call = mod.get_global_var('func_5968')
func_5970_call = mutated_mod.get_global_var('func_5970')
call_8154 = func_5968_call()
call_8155 = func_5968_call()
output = relay.Tuple([call_8142,call_8154,])
output2 = relay.Tuple([call_8143,call_8155,])
func_8166 = relay.Function([], output)
mod['func_8166'] = func_8166
mod = relay.transform.InferType()(mod)
mutated_mod['func_8166'] = func_8166
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8166_call = mutated_mod.get_global_var('func_8166')
call_8167 = func_8166_call()
output = call_8167
func_8168 = relay.Function([], output)
mutated_mod['func_8168'] = func_8168
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8121_call = mod.get_global_var('func_8121')
func_8123_call = mutated_mod.get_global_var('func_8123')
call_8188 = func_8121_call()
call_8189 = func_8121_call()
output = relay.Tuple([call_8188,])
output2 = relay.Tuple([call_8189,])
func_8193 = relay.Function([], output)
mod['func_8193'] = func_8193
mod = relay.transform.InferType()(mod)
output = func_8193()
func_8194 = relay.Function([], output)
mutated_mod['func_8194'] = func_8194
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5642_call = mod.get_global_var('func_5642')
func_5644_call = mutated_mod.get_global_var('func_5644')
call_8213 = relay.TupleGetItem(func_5642_call(), 1)
call_8214 = relay.TupleGetItem(func_5644_call(), 1)
output = call_8213
output2 = call_8214
func_8235 = relay.Function([], output)
mod['func_8235'] = func_8235
mod = relay.transform.InferType()(mod)
mutated_mod['func_8235'] = func_8235
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8235_call = mutated_mod.get_global_var('func_8235')
call_8236 = func_8235_call()
output = call_8236
func_8237 = relay.Function([], output)
mutated_mod['func_8237'] = func_8237
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8139_call = mod.get_global_var('func_8139')
func_8141_call = mutated_mod.get_global_var('func_8141')
call_8248 = relay.TupleGetItem(func_8139_call(), 1)
call_8249 = relay.TupleGetItem(func_8141_call(), 1)
output = call_8248
output2 = call_8249
func_8253 = relay.Function([], output)
mod['func_8253'] = func_8253
mod = relay.transform.InferType()(mod)
mutated_mod['func_8253'] = func_8253
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8253_call = mutated_mod.get_global_var('func_8253')
call_8254 = func_8253_call()
output = call_8254
func_8255 = relay.Function([], output)
mutated_mod['func_8255'] = func_8255
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6816_call = mod.get_global_var('func_6816')
func_6817_call = mutated_mod.get_global_var('func_6817')
call_8382 = func_6816_call()
call_8383 = func_6816_call()
func_7940_call = mod.get_global_var('func_7940')
func_7943_call = mutated_mod.get_global_var('func_7943')
const_8385 = relay.const([4.540082,-3.592865,6.581627,-1.901775,-8.841155,-6.762177,3.300100,4.450198,-3.515907,3.791144,6.169144,2.238128,0.996547,9.635950,-8.544748,-1.631641,7.459310,1.036042,-4.106534,-4.315330,-8.863416,8.175616,-4.238316,-9.542956,1.025745,1.282227,1.082533,2.152896,3.169939,-3.143127,4.535917,4.138967,1.496591,2.149540,-9.083281,-0.804149,0.640950,2.684137,-2.350763,-8.950463,7.097113,-2.214094,-6.677651,3.165306,4.579141,7.283493,-6.236106,2.864561,-2.466955,-5.065820,-4.383583,1.290296,-5.302742,-2.527962,-5.595266,-4.415090,-2.051140,5.656434,4.323449,-0.668285,-2.869867,2.900229,-2.481642,-1.170242,-2.723525,3.916344,2.895679,-9.255357,3.286221,-4.084059,0.112170,3.054476,-9.910334,0.263068,8.029770,-4.787675,6.066101,8.429602,-4.482799,-2.314884,-0.321826,7.123079,-1.786287,0.068755,-8.704022,-8.382125,-1.447803,2.016160,-1.858920,-0.453507,-4.139007,9.674835,9.613382,3.867022,-4.858213,-3.762283,-9.435392,-1.291003,0.423306,-3.292025,4.507064,4.161829,3.196731,-9.802873,7.206306,2.979548,9.150879,-5.310945,3.048903,-6.014015,0.012793,-4.222400,9.437555,4.310024,-6.365270,-6.706071,-8.924557,-6.680127,1.458358,9.833160,2.132232,-2.734622,7.003994,2.410528,9.227750,-5.473102,-6.582810,-8.288790,3.520819,-7.098297,7.154642,6.562409,9.373679,8.651950,5.509438,2.761760,-0.405516,5.466833,-1.188606,-3.469393,3.734545,1.972305,-0.910718,5.757243,1.326529,2.526806,-9.216921,-9.779336,-5.183638,-4.716532,-0.582967,5.320546,-5.185639,-1.483934,9.303890,-8.302313,1.686192,7.946970,4.753992,5.598380,-1.013078,2.090453,5.565222,-9.344258,-0.595240,8.026346,7.405068,7.986158,-9.333379,-2.224664,-9.241450,-6.487298,-6.519089,7.759695,-9.866352,-1.982970,-6.624226,9.102229,1.840212,8.520479,5.299436,-3.910847,6.834447,-5.946176,9.116582,-9.454884,7.681126,4.840017,-4.503189,1.285120,3.797262,9.244064,7.114548,-5.437960,6.138807,-7.087326,-1.255039,2.222572,-5.997213,3.166010,-4.509754,-1.753678,-6.776895,-8.595346,-0.298699,2.618634,4.293802,8.440910,-7.309019,-4.754564,-9.648610,-4.777476,7.312263,-8.533662,8.652762,-3.910372,2.085648,5.223534,-4.622507,-5.529003,3.614390,4.153626,6.538338,6.988017,-0.350109,6.967765,0.936689,6.545174,9.841576,-5.797058,-3.307099,-7.518574,9.783079,5.610171,5.811506,0.424662,-8.222963,-6.261269,5.959373,7.275745,9.840104,-1.506313,-2.942956,6.593797,-9.011728,-3.954934,-5.295554,-5.085209,-4.989945,-8.096086,-2.575061,-4.636107,-1.246082,-0.271321,-5.557458,0.622954,-1.674647,-8.601499,0.803588,-0.495658,-9.800901,-3.080059,3.936862,-2.890957,8.981526,-3.523743,-6.354725,-0.630903,1.807863,7.011420,-1.099308,9.169165,-4.444820,8.878660,5.620611,1.839438,-9.030045,3.979670,5.772325,0.567904,-0.953067,-1.396648,-8.576255,6.445075,0.311039,4.130352,3.760581,-9.620210,4.084306,5.836134,9.538729,-7.423563,6.871400,0.916820,4.479109,7.124970,-0.609344,2.338400,-1.937346,-7.532441,5.672852,3.259516,-3.838401,-8.997442,3.213205,8.850545,5.454199,-5.197503,5.223709,8.148007,-4.527493,-7.808380,-2.363921,3.253113,-8.458957,-1.185220,3.689548,-9.164109,7.773882,4.019500,-7.260326,2.410524,-4.268902,-9.074144,8.090225,2.101714,-7.928424,8.229925,8.134947,2.817849,-5.148971,4.937010,2.081067,-6.744970,-7.331569,8.035192,-2.143216,-8.289327,-7.167341,-2.171720,-0.543344,0.791297,6.598123,3.514898,6.222772,-1.762013,-1.927408,-4.730639,3.733598,-7.266351,-3.795550,-1.207222,-8.245529,-2.495321,-4.394021,9.451941,-9.147965,7.176174,5.542864,-8.488033,-0.784296,-2.308137,-8.248828,-1.122763,7.738041,8.257917,0.389937,-0.819801,3.233344,-4.588508,4.580894,-7.911918,3.130565,3.397702,8.320164,1.420355,-4.433278,-1.888619,-4.917756,-6.665627,-1.107327,-4.438491,-8.929963,1.799207,5.567718,8.298518,1.598358,-2.806862,1.979720,6.632338,2.287924,-3.558077,2.958449,-3.899947,-5.613117,9.503725,-5.959237,-1.409765,2.307535,-6.095600,8.345439,0.493250,-0.607187,8.623034,-1.934655,-4.353811,-8.878027,5.388096,5.726549,9.238029,7.134711,-2.043250,-9.498553,-8.631937,1.392903,-2.543871,-9.476385,9.166616,4.465985,-5.224137,-7.799593,1.991728,-9.846589,-7.214819,-5.973301,6.738644,-1.823553,-1.588409,-2.288689,-3.129431,7.214869,-1.932455,9.533633,-0.502524,2.872843,-2.666302,5.927551,-8.230110,-2.878838,-0.565144,-2.476642,-5.315664,6.186461,-8.563525,8.783796,5.276500,6.304051,-3.222465,-1.644628,-3.471547,-5.176190,4.717776,-4.145610,4.724894,-7.224347,-3.080369,6.345189,8.041269,2.864521,-3.063463,4.652153,7.967794,2.284011,6.410058,4.638094,4.042563,-3.381902,0.486796,-2.772395,7.936900,-2.101841,-3.798847,1.934831,6.650882,6.523733,1.254503,1.184876,6.172371,4.332156,-6.354370,-3.714530,-4.997940,-7.861090,-5.990636,-0.984012,0.067942,7.591431,-7.158761,-5.368270,5.989875,8.323879,-8.013939,0.472766,8.874594,-8.202946,9.237916,-0.576512,4.971201,-4.722300,5.866067,9.256545,8.296960,-2.743593,4.506959,8.267009,-4.012994,-7.383931,-5.596279,7.885356,-1.743765,7.322161,-9.838040,0.283832,5.802477,9.996707,-1.948396,2.833992,9.478581,3.094402,-9.875337,5.306664,5.399837,1.227576,-4.106108,4.375187,-6.229924,-0.173842,-0.273770,-8.994269,4.033745,0.905688,1.682057,-4.558515,-8.967542,8.572258,-0.939826,-5.264401,-5.079862,0.618374,-7.106818,5.678659,-6.656082,9.698090,4.647567,-7.663147,-3.633330,-6.933047,-2.174738,-0.589734,-1.399890,-0.978620,4.471165,-4.392301,-9.821000,-3.132225,8.145988,-0.049409,0.741283,-0.257557,2.621220,-7.375551,-8.427160,0.648121,4.412852,-9.300117,-4.929356,-6.260015,8.450961,1.870114,8.629757,-0.958903,-1.864477,-9.704928,-4.088066,-7.009337,3.457025,6.037262,-7.689665,-1.923050,8.098237,4.042694,5.425993,4.324067,1.895275,7.673338,-5.496851,0.897503,-7.528001,-6.067915,-4.596395,-2.711959,0.377568,-3.281961,0.774331,5.211009,7.370922,-8.029031,2.046805,9.404145,-2.410043,7.475331,-9.763153,6.798376,-5.625659,8.636682,3.185229,-4.858369,-3.160492,-9.680448,2.842915,-6.243593,-8.194523,-7.971722,-8.322972,-4.424205,-4.337527,7.026658,-1.515931,2.763568,-1.820643,-2.085837,-2.797308,1.685510,9.199748,-6.196809,4.885438,7.179625,9.756626,6.483537,6.101754,1.562682,4.704940,-1.575917,1.041171,-1.733782,-7.893945,7.315230,8.253027,9.539579,-1.994186,-1.246762,-4.583950,-2.927757,4.192225,6.269860,6.352779,-2.839267,0.724747,8.260143,-1.381742,7.634344,-6.700150,9.799882,2.178298,-0.860734,9.403348,2.508352,-5.564786,1.953889,-4.097105,-2.324553,1.368932,-7.123143,4.378482,-6.575859,-0.024718,-4.153496,-3.081933,-6.571313,2.965236,-5.515923,9.001606,-6.401781,2.564751,5.142805,4.576302,0.491340,-1.839829,3.289975,5.903307,0.499222,9.409702,-5.187423,-6.771829,8.737758,5.689512,-7.616934,-4.645851,-2.323056,-8.560003,7.959616,4.087582,6.864437,-9.283222,3.508839,-4.621561,7.366181,2.891635,7.925713,2.112752,7.078697,-9.664980,8.321045,-0.151507,-2.480750,-1.212639,-3.066478,8.352251,5.405721,1.456655,6.574095,-4.575991,8.347846,1.794840,1.535580,-6.142688,2.788554,-3.719867,0.828501,7.356850,4.394181,1.929382,3.902356,2.079072,-4.197355,-8.430196,-9.345589,4.865409,-4.645988,0.737195,4.556846,-6.799158,-1.872360,1.065699,4.187064,-0.415068,-1.671820,8.607920,-1.298483,3.227545,1.789030,-9.877439,-2.342820,6.007830,-7.815107,-8.582577,-4.224855,8.410042,3.263649,-0.158805,4.362273,3.565667,-5.512017,8.097682,-3.727315,-9.591204,-8.531103,4.811748,-1.519704,3.255910,9.806078,-4.145472,-5.594346,0.937313,-9.490994,6.843655,-6.645810,-1.000836,2.336265,9.956716,-6.778023,3.887275,-3.476222,2.081248,0.479018,-9.012465,-8.421719,9.624926,-0.166741,-7.113555,-0.291316,-5.734302,-6.934109,-7.460196,-5.588572,6.709335,4.685455,-9.738942,-2.294703,-7.761541,-1.296375,-9.671572,3.662982,-4.223322,-0.885661,-7.360813,-5.196558,6.480870,1.145810,-4.541490,1.040043,-4.946431,-3.068612,8.310660,-4.999147,-1.124721,-3.054532,5.727695,-7.332631,0.136652,-8.965767,-0.974871,-6.599453,-8.268728,9.212561,4.844558,-1.078758,3.736407,-5.014360,0.217080,-1.990792,-9.374613,0.325255,-4.665726,9.901232,-7.050647,4.733517,-0.994974,3.964992,-0.476384,3.347801,2.189636,8.526401,-6.425486,6.762693,1.470691,8.005101,-4.665183,5.530127,0.775627,-5.657570,1.686127,-6.795581,8.721300,-8.547239,-8.721841,-0.576108,2.209549,-9.724337,3.762962,-2.710534,7.024853,-8.764195,-2.426417,-5.412088,-9.077774,1.278135,-1.683261,2.435670,4.582159,7.191762,8.866228,-7.987777,-3.059325,-1.852193,-7.480811,-9.469977,1.004443,-6.446887,-0.436568,2.110245,1.798156,-2.628010,-2.327396,-3.222814,0.346655,6.616454,8.798410,-5.498123,-4.893127,9.827151,5.376410,-2.377373,5.764511,-0.222234,0.404730,-8.462552,-8.001414,-6.745311,8.314165,6.227475,2.410107,0.955708,-7.530905,-7.780781,8.342701,6.782277,0.276741,0.999587,-0.980298,-9.703747,5.148550,8.110442,-3.807239,-0.107002,-5.826465,-6.819145,-4.816350,-1.658093,3.349719,5.350358,-6.418111,3.984681,-9.100010,-5.172703,-7.847597,-4.697434,-6.183790,-0.215956,1.142293,-6.434274,-1.163694,8.927857,-3.440920,3.685762,-6.954229,-6.648882,1.503399,-4.664631,-3.206007,7.522531,-1.593607,-0.319789,9.281112,8.844733,7.492459,-3.462326,-6.713610,-9.428968,7.812986,1.518232,-5.659107,-4.727600,2.333212,0.129682,-9.711796,5.832141,-3.524241,6.071651,-4.909688,8.827856,-4.366703,5.504832,8.802834,3.536464,8.823528,5.152684,2.687225,9.190573,-2.481268,-3.848261,8.702385,1.402803,-3.081339,4.796488,8.757552,-6.816169,-7.788372,9.318371,-5.465318,-6.426925,4.653341,8.878003,-7.532128,1.639939,-1.274488,0.564316,9.080403,1.849546,9.940922,-6.008290,1.661322,-2.737541,-2.684722,-2.015398,-4.168012,-0.424253,-5.117347,-9.566941,7.056413,4.405346,-5.830793,8.292645,-3.387020,9.893750,-9.006332,-0.789663,-9.323215,1.962551,-1.403412,7.582132,2.349811,-3.925162,-8.637151,-8.967903,2.484199,8.206307,-1.949457,0.650896,3.130123,-7.725126,3.012713,-8.498802,0.580414,-1.333896,2.655266,-9.699634,-6.126008,-6.307776,-5.541239,-1.747650,-8.590271,8.632733,-5.266745,6.067315,7.646267,0.481389,1.435369,-0.573818,-5.368728,-1.180837,0.023249,-5.478254,-4.441734,-9.399546,5.842611,-6.912956,-9.569895,-3.284914,-3.771398,-0.624814,1.718861,-5.075613,3.095772,0.587408,-0.058085,1.933413,-7.779712,4.381969,-5.657975,1.605144,-0.792336,-1.134801,6.876449,1.082530,-0.517138,7.172104,-2.572201,-4.905372,8.398977,5.263240,-8.336639,4.286575,-5.618407,8.485672,-1.018919,-7.023813,-4.733504,3.219975,0.313232,9.574501,-0.712825,9.018277,3.912934,3.369838,7.851836,-1.502976,0.244317,-2.795252,-9.082376,4.560223,8.582461,6.389681,-0.188116,6.826275,5.722216,1.522948,-5.660083,-2.615235,8.339456,-9.781987,-1.123876,-5.274316,-0.823279,8.227073,-6.484674,9.638391,-6.682079,9.661241,7.440590,-8.687334,1.095538,-1.822800,-7.805686,7.591017,-2.781878,-7.535903,-1.295570,4.085415,-1.005145,6.396257,2.118430,-9.462401,7.200311,-3.430283,-6.917386,-4.958506,-7.251444,0.705214,6.168061,-7.275197,-0.773773,2.798419,9.406289,9.797915,0.123211,-6.324708,2.815657,4.087100,-1.957727,0.097466,-8.869718,3.388198,4.533754,-4.084639,1.993000,-8.357171,5.717191,8.920518,-9.211439,-1.303666,0.893148,-8.079388,1.199730,-1.755475,0.976411,6.412699,6.663057,9.815529,1.628834,9.425149,-0.298214,4.811139,1.964288,4.078740,7.959632,-3.294105,-9.522219,-1.261199,-8.091316,-8.460261,-6.472845,5.334342,-3.986366,-5.690957,-1.215411,-4.770441,-7.408508,4.690073,-8.024441,9.413166,-1.410124,3.988435,1.793499,7.914971,-4.822639,4.369254,7.043386,-1.691864,8.839862,8.749125,-2.164842,1.631079,3.890340,5.254496,9.084577,-5.514159,-0.093301,-1.457665,4.154361,8.677492,1.850461,-4.504760,5.366850,-9.885999,-6.851782,3.080229,-9.793190,2.283353,6.882095,2.882378,1.728717,-6.034749,2.402037,9.858058,-4.390195,-3.614317,-2.549776,-9.122745,6.760885,-8.908767,1.216371,1.491918,-1.030390,9.249681,0.960443,-0.740899,-5.291422,-1.324457,-3.196362,-8.610762,-0.482824,-6.469312,-7.496196,-5.816915,-0.005902,0.216350,-4.288037,3.276433,8.214183,-8.986519,3.209090,0.409145,-7.754100,-2.207421,-2.380947,3.450065,4.114648,0.684149,9.521291,-4.462188,-1.459780,3.559039,-3.070371,-3.199061,5.672881,-6.252919,6.826927,3.182141,2.604483,-7.429749,-4.910279,-7.913989,-8.298129,2.802445,3.053780,5.071566,8.087705,9.295103,0.334127,8.240536,-5.868061,7.425148,-8.772852,9.154175,-2.164493,2.411570,-4.891274,7.812738,7.250814,-9.106716,-4.614239,-4.258994,-6.420145,0.945248,9.480601,-5.466620,-3.010382,-2.500770,-1.668146,-3.526628,0.223147,-4.347924,5.993008,-5.402356,-7.629495,9.912420,-5.860974,8.546156,0.919662,-9.833802,8.761894,2.750483,3.639321,-3.282667,0.041785,-9.237235,-2.255897,-9.249600,-9.162601,-9.521660,1.844229,2.888797,-4.261837,-0.990694,2.708557,-7.698483,-4.447566,-8.712080,7.986257,4.884857,3.087003,-3.230463,-8.846385,3.295991,5.634734,-9.631656,-0.993289,7.446177,-9.492418,-6.646225,-2.488643,-4.305184,-8.515574,-0.521632,-7.017880,1.014026,-7.148557,8.791612,-5.081766,7.156388,4.615010,0.517323,-5.669054,-0.950675,-3.653847,-9.055025,5.296772,8.282304,-6.435387,-9.147534,-3.643076,-0.179896,4.434392,9.617380,3.198995,6.672200,-8.035883,-8.517169,1.875883,8.648410,6.873810,8.716518,8.695926,0.211346,-7.868748,-1.332075,6.396168,7.032618,-1.832525,-3.725821,-9.521903,-6.889849,-8.997769,9.394248,5.417267,-2.238909,-2.643581,-6.359794,-1.908802,0.334906,2.586224,4.056357,8.142720,6.700540,-3.895451,1.385010,8.443118,5.203471,-4.888591,-9.467084,-4.912932,-3.062443,-9.436772,-0.811468,-7.287133,1.150796,3.806842,-3.813889,-2.601143,-1.381648,-5.828919,6.029828,-7.294056,5.607929,1.134712,-7.792617,-8.594579,-4.600052,7.056716,-0.636707,4.308505,9.698332,-2.952737,5.983464,1.476485,7.600605,9.496635,-2.594830,-2.562463,-5.468446,8.355014,5.469607,0.329820,-0.135192,-4.869898,7.080879,-1.771609,-3.808584,-8.838644,1.797878,2.760024,9.132965,4.903231,-2.969148,-5.801896,9.918485,-6.559397,0.335498,3.073354,-5.054377,-3.989648,9.804833,7.639217,-8.099603,9.884601,3.851551,-3.197026,1.530284,-0.196663,1.369495,-7.743091,-4.395856,4.860356,-6.853811,3.743456,-1.839576,-7.838975,2.696808,7.114082,-5.867176,8.202361,-1.144657,-4.740031,-9.511211,4.267852,1.471306,-9.063745,-0.531248,-5.230746,-1.511094,-7.957977,6.214783,-1.746344,-9.152585,-3.149130,-7.761566,-5.859746,-7.453105,-6.249612,-5.232887,-4.258629,4.593027,4.443460,8.561916,3.380257,-3.563241,-2.767267,-4.575534,-3.441445,3.988800,1.117580,-1.830364,-9.483307,9.717592,5.528754,-1.517784,0.230296,4.299054,-1.679996,-5.436880,-7.057688,2.215416,0.059797,-9.292427,-2.473056,-2.058951,5.080788,7.261456,1.771466,5.811723,-0.272481,7.628933,4.466494,4.669121,5.867728,8.833613,-7.312529,-8.306060,6.260296,9.413817,-4.539356,4.116169,-7.002320,8.771803,1.843165,5.088055,3.894697,-9.753854,7.373895,-1.907283,4.079727,-3.134516,-7.159405,6.571924,3.411790,9.776848,8.239664,-7.305922,1.221589,2.016526,-3.637339,7.524084,4.076222,-8.280873,2.967583,9.991378,3.543216,-4.157253,0.052825,9.430689,-8.483704,-1.355030,4.960734,1.998355,4.326800,-3.126814,6.743200,2.170353,0.204288,2.196153,-9.284561,5.609389,6.886562,1.946067,-4.206704,3.617330,6.897765,-1.170467,-9.713466,7.705181,-7.971416,4.305233,4.976101,4.249514,3.925649,1.316534,0.500095,1.757350,-6.491590,8.370746,5.530931,6.208764,7.476190,0.419941,-7.022830,0.849790,-3.154197,-8.134535,7.833469,-6.892438,5.342127,-8.836606,7.699079], dtype = "float64")#candidate|8385|(1584,)|const|float64
call_8384 = relay.TupleGetItem(func_7940_call(relay.reshape(const_8385.astype('float64'), [11, 16, 9])), 0)
call_8386 = relay.TupleGetItem(func_7943_call(relay.reshape(const_8385.astype('float64'), [11, 16, 9])), 0)
output = relay.Tuple([call_8382,call_8384,const_8385,])
output2 = relay.Tuple([call_8383,call_8386,const_8385,])
func_8388 = relay.Function([], output)
mod['func_8388'] = func_8388
mod = relay.transform.InferType()(mod)
mutated_mod['func_8388'] = func_8388
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8388_call = mutated_mod.get_global_var('func_8388')
call_8389 = func_8388_call()
output = call_8389
func_8390 = relay.Function([], output)
mutated_mod['func_8390'] = func_8390
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8235_call = mod.get_global_var('func_8235')
func_8237_call = mutated_mod.get_global_var('func_8237')
call_8416 = func_8235_call()
call_8417 = func_8235_call()
output = relay.Tuple([call_8416,])
output2 = relay.Tuple([call_8417,])
func_8428 = relay.Function([], output)
mod['func_8428'] = func_8428
mod = relay.transform.InferType()(mod)
output = func_8428()
func_8429 = relay.Function([], output)
mutated_mod['func_8429'] = func_8429
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5200_call = mod.get_global_var('func_5200')
func_5201_call = mutated_mod.get_global_var('func_5201')
call_8440 = relay.TupleGetItem(func_5200_call(), 0)
call_8441 = relay.TupleGetItem(func_5201_call(), 0)
func_7442_call = mod.get_global_var('func_7442')
func_7445_call = mutated_mod.get_global_var('func_7445')
const_8443 = relay.const([5.524016,-0.405117,1.204508,9.880118,-4.664157,-3.889609,4.070990,9.713618,4.136762,-6.440442,-1.104222,-3.564017,-4.819894,-7.957470,5.857426,-4.259122,1.504891,3.215634,-0.950316,2.606855,-0.954753,5.077897,-6.508135,-5.117071,-8.442373,0.695700,4.873692,7.308694,-4.836729,-5.997152,7.307399,6.064218,0.960204,-9.957843,3.192973,-3.240360,3.567278,1.970461,-2.982442,4.018629,-9.899263,0.029869,-5.486809,6.117824,5.732438,-5.482685,0.677952,3.484179,-1.022213,-5.305222,1.922929,1.608985,-1.464396,9.235762,7.679722,8.258220,3.463411,2.555382,-6.785580,1.631025,-0.669756,-7.624531,0.264376,-3.139692,5.058094,8.318011,-6.915111,2.164149,6.888920,9.514945,3.494210,-1.954638,-2.868761,-4.019633,0.751183,-3.569341,3.222014,0.460671,-5.474938,-4.292641,-6.032913,4.670698,6.918270,8.181108,6.166004,8.002664,-6.969013,0.654478,9.301045,3.759793,2.666290,6.566782,-5.922789,3.480929,-0.975095,0.689634,6.390812,1.197466,6.756834,3.091239,-5.721782,-5.485824,9.389511,3.904290,3.938855,4.045563,-0.451362,-3.848383,9.529201,7.534804,9.779264,-3.141283,5.190210,8.370667,-1.834364,0.806251,0.176391,0.290227,-9.156124,-3.772066,6.298254,-1.330958,-0.010920,2.877562,7.415265,5.422053,5.899212,3.471768,-9.968093,0.765993,-5.093487,1.598706,-9.473654,-6.200877,1.178078,8.662613,-4.283248,-3.357179,-0.355670,-9.653466,-9.381017,-4.421096,9.432502,3.435780,9.371610,-1.035527,2.518053,9.970229,5.271615,-4.384591,-9.629724,9.665654,-5.616242,8.686508,7.430222,-7.004694,-6.363567,4.952791,-1.899439,9.327314,-2.837402,1.408515,6.292325,3.928424,-2.981107,5.334075,0.758050,1.210884,-6.468619,-8.009274,-3.822986,2.196967,-6.667796,-7.442477,3.751152,-3.253188,-3.364312,-9.408922,-5.920722,-0.701378,4.411728,-9.816749,-8.010623,-3.124506,-6.648474,3.914893,3.936180,3.864931,3.512620,0.925816,6.668750,-2.320625,7.823187,7.133508,0.008951,2.288432,8.392448,3.544128,1.524326,-5.816881,2.024005,0.036434,4.151982,-8.615978,-7.220950,9.146836,6.088145,2.870833,-2.143735,-5.313347,-9.634502,0.015249,7.119685,-8.396891,-4.877223,2.169220,-8.130008,6.865844,1.964082,-8.173264,4.861545,-6.896973,-9.980472,6.887338,-7.885582,-7.465853,8.939294,3.481024,-4.728058,-5.244911,-4.929741,-8.884109,-3.650498,-2.418894,9.471769,3.159780,4.914682,9.248301,-2.058530,-8.545972,3.770673,5.195892,-0.908677,9.522555,-5.840105,-7.163636,7.330172,5.051106,-2.443707,-0.479839,6.979154,-5.903346,9.666252,-3.352103,-3.551259,-7.073568,6.449860,6.376809,-2.950244,-1.817052,-9.804387,-4.806239,-6.763134,2.301525,0.912494,-1.923082,5.398871,6.366715,2.601502,-9.564258,7.229559,1.432007,7.811086,-4.772657,4.137878,-3.671003,8.997461,-6.208796,6.709612,-6.989783,6.663461,3.849152,6.788440,7.588397,-9.132465,3.571065,-3.632669,-9.868459,8.677286,-0.239928,8.202348,-5.060357,3.001909,-9.841442,-0.362282,2.939547,-1.547074,-8.425610,-0.267363,-5.566796,-6.305394,-3.779382,-6.653497,-5.198647,5.652114,-5.974479,3.805696,-7.150478,9.779346,9.201165,9.614766,0.474681,1.351581,-0.036298,3.618491,-9.931819,-2.649147,8.504793,4.644654,-1.533138,-5.858492,-7.470607,7.296103,-4.219539,5.762396,-5.495425,-7.525790,-1.753650,4.166691,5.182468,6.000077,-5.677952,-9.529980,-9.620718,3.193218,2.504317,-6.382066,-1.377206,-2.933425,-1.615385,-0.278120,-1.120510,7.150024,9.783005,2.946329,2.940632,-2.116018,-3.728588,7.763200,6.176315,5.395293,-4.719876,-5.163594,6.174542,5.536847,-8.348528,-1.842776,6.197860,-9.537808,9.266797,3.370544,-5.324971,-2.139564,9.587407,-5.152861,5.699486,9.609998,0.162003,-4.030970,-8.375216,-6.318263,7.486632,2.549843,-6.843153,7.349732,-1.241344,7.968546,7.761057,8.248314,6.845480,2.848013,-8.937402,5.879591,-2.543055,-8.792772,1.347400,6.596062,-4.365545,0.163000,-3.130730,-1.490719,-9.349492,-1.458196,8.691792,-1.767077,2.257088,3.367876,-9.791273,9.336704,-8.828452,3.857008,-2.040819,1.961173,8.763246,-4.173024,8.651526,-1.421031,-5.401390,7.497824,4.397935,-1.281803,9.755783,-6.363377,-8.418522,-9.851415,4.534287,6.958015,1.551834,-4.650814,1.197569,-3.346807,0.919360,-7.574437,0.962713,-1.833757,4.014202,7.397882,-3.980101,-2.163768,8.228291,-0.712812,-0.828060,5.606157,-4.725524,2.157307,-3.002049,-9.489266,-2.685576,5.681178,4.372194,-1.096979,-5.753666,8.834690,0.703483,-2.087080,0.520957,-2.553247,-5.239988,7.231167,9.596469,-5.882244,-3.774666,-2.485076,-9.335174,9.400694,9.315574,-7.312008,7.849573,-2.275378,-1.406474,-4.327237,9.199911,2.737042,-3.300723,-0.515127,9.332728,5.236586,-1.810809,9.417331,2.397197,-3.373364,-3.380507,-4.837400,-0.514393,5.085844,1.939944,-7.261338,-0.312697,-0.273958,-9.680558,-8.508025,-9.878231,-8.825559,2.555883,5.417613,-4.021793,1.818059,-0.475115,-3.502644,6.347918,2.590000,8.536196,3.677726,-5.153961,-7.266519,0.744694,-3.629006,-8.704937,-0.512011,-1.146826,-0.180675,6.811780,-3.861209,-7.368253,-6.459331,-8.775083,-3.580029,-1.430945,7.498097,-8.990894,0.512791,-5.505714,5.413588,-1.709541,8.567106,6.478609,1.707067,4.367548,-2.519614,6.725011,9.065962,4.066217,7.519707,-4.771393,-0.839428,7.087178,-6.426403,0.363097,-9.723309,1.287441,3.645806,-6.903346,4.195794,-1.886096,3.113106,-6.865952,0.854868,-1.276080,0.590566,0.177319,3.996233,-6.078894,-7.167880,3.512787,8.852226,-8.888800,3.225609,-3.814741,8.221339,1.333051,-4.340131,0.961564,3.152846,-8.694910,-2.184467,-5.866707,-3.917380,8.412866,-1.339263,-4.778946,1.087833,-0.064923,4.958510,7.690563,5.741094,0.631128,2.070538,1.089122,-1.003346,3.348196,3.320797,8.647718,-0.183237,4.207326,3.765105,2.660089,-5.261553,-1.759911,-9.257775,-1.925159,-6.874341,7.808050,4.726707,-4.505863,-7.075597,-4.987275,6.251677,-0.539557,7.501504,-8.976031,8.722242,5.821772,-3.425733,0.970436,-6.472912,-0.206212,1.737168,-3.646509,4.926333,-2.055894,-3.779460,6.851695,-9.296430,-3.816662,8.667775,7.819354,-3.031091,5.709529,-8.872202,3.478015,-7.410880,4.324562,7.593313,-5.745022,-1.341179,-4.002447,-9.127985,-2.971776,0.795650,-1.762796,2.224662,1.370484,-4.885021,-8.495267,9.163960,7.044209,-8.500176,8.345505,-0.134020,-1.922430,7.830936,-2.094883,-1.853599,-9.570404,0.339860,0.328741,-5.239839,-1.426251,0.765065,5.875410,-2.515011,-7.168516,2.110953,-8.664568,-6.797185,-6.731125,5.851955,4.671966,0.216921,-8.049261,-3.827086,-3.385883,-6.978439,3.268904,9.188012,3.449196,-9.866994,-6.299499,7.637675,6.881273,3.210680,2.655862,-4.532577,8.020825,7.247156,-5.450459,4.102246,-9.056023,2.541744,3.948303,2.494636,-9.339689,-6.238384,1.625368,3.641469,8.831071,3.249303,-0.848713,3.198711,2.006434,9.592716,-7.975441,-2.394755,-7.380396,1.329180,9.813398,-7.559921,-3.935573,-6.589698,4.301771,-0.177588,7.431055,0.754681,-7.768551,1.807119,4.832606,3.690510,4.896578,5.417956,-8.934767,8.614826,1.942566,5.428727,-6.693442,3.009680,-3.066317,2.708984,3.204634,2.677870,9.948548,-0.630575,3.573294,6.176950,0.696066,-1.796433,8.890355,6.970764,-8.609691,-3.815016,8.162320,-2.969298,6.034381,1.310760,-7.837658,-7.118444,9.782305,6.171337,4.172115,9.054118,7.186094,-1.813925,3.279188,2.395726,-1.230999,-0.937226,5.342752,2.954618,-1.865060,-2.232942,2.655730,2.248461,4.025023,9.930715,9.969525,8.939076,0.592932,8.929501,-2.317345,-9.292102,-3.060010,-0.938710,2.223208,-5.577280,-4.379264,-6.857618,-8.048739,-2.390958,-4.199553,-8.918163,-4.540378,-2.885909,8.891325,-2.422675,-7.434350,-8.662408,3.795744,5.222734,3.964454,2.483272,-6.838479,3.652115,2.046405,-6.686061,-4.074665,4.260167,-7.098353,-1.059106,7.729463,9.979411,6.048056,-1.326298,-3.200879,2.297207,2.796301,5.399308,-0.299694,-8.763689,-2.766152,-7.751362,2.998601,7.432280,9.615966,7.064954,-1.889271,7.082534,1.950323,2.820573,9.331605,2.540547,5.142899,-1.961452,8.201564,6.661047,6.836669,-9.914708,7.162079,-6.233180,2.221893,5.559030,9.697716,5.441289,-5.800680,-7.700623,8.158813,-7.837335,9.810956,0.470059,-5.828579,-6.142463,-9.291205,-2.922105,6.197593,-0.491961,4.925974,-3.216423,-4.623900,9.144451,-4.042140,-7.734668,8.002682,-3.365577,6.193591,0.211454,-0.876393,4.505773,5.235955,-3.216601,5.862666,4.259336,-1.828976,7.012050,-0.522297,5.401476,-8.251391,-8.027331,1.840774,-4.063666,-3.370216,-8.105633,-2.117214,9.362510,6.036662,-4.304178,7.718853,4.305682,2.455958,-2.527373,2.044780,5.546784,-9.912602,7.895322,-2.514108,-5.263509,-6.077370,-2.958982,6.560940,-4.502259,6.849687,8.873345,-1.295878,5.254952,9.341942,3.636700,8.406565,-4.806213,2.798944,6.533061,-5.699063,-1.298629,-0.035968,5.477826,2.808486,7.082463,-6.757873,8.166982,4.602480,-0.055044,-4.344427,-2.890642,-5.331125,9.858138,-9.834590,-2.119250,1.528669,4.686817,1.807161,7.523572,-4.075766,7.283597,1.623080,-3.118170,0.440768,8.174154,-2.619945,-2.224069,-9.578879,-5.746409,-5.634068,8.937814,-5.174702,-6.477899,0.760676,-4.332802,-1.985788,-9.655676,-1.801757,-3.707657,-5.195592,8.214733,2.198102,4.336590,-6.255140,6.564779,-8.661453,-5.271669,7.288918,-4.311407,-5.534837,8.627830,3.539192,-4.531962,-5.396113,6.489964,-7.464596,9.081141,-7.491845,9.429077,5.575076,-6.964887,-1.626629,-3.482850,5.999381,-6.022877,1.693221,-4.529164,-2.346255,-3.226598,3.700631,5.167189,-5.089979,-3.257817,0.905802,9.968742,-8.068068,9.130515,-9.078694,-3.381090,-7.499443,8.332786,-2.382056,-6.014759,-5.310181,2.416448,0.724117,3.318332,-5.324441,-6.806402,-1.170218,-4.768533,3.000667,7.899067,4.112370,4.663747,-6.504888,5.440679,-9.893381,9.529466,-7.019887,8.665718,7.532782,9.771604,-1.059106,-1.284507,-7.377876,6.796784,-1.243630,6.813059,-1.132703,2.298537,-4.103766,-6.246570,1.667030,7.748293,-8.731903,3.780509,-0.830201,4.646828,-6.056886,9.510509,-6.547222,6.398393,8.546779,-2.734661,0.288825,-0.130417,-5.381472,5.324251,2.109684,5.227652,6.576909,8.556923,-2.935369,-7.596009,-1.428011,9.510658,-5.636011,7.714786,3.812253,5.690377,-8.958515,-0.271576,-1.215033,-1.623234,1.426514,-8.983785,-6.077369,-1.507973,-9.900928,4.789117,-2.423584,-9.894155,1.147379,-2.595392,0.560406,-5.373220,2.118579,0.913809,-0.243747,5.893615,8.114248,-6.776563,-3.215157,-2.882712,4.725834,7.997130,-5.306000,-0.341984,-4.725820,2.031780,-7.014085,3.843106,-5.002598,-4.348050,-7.603092,-4.702831,2.760882,7.882164,2.711643,-8.808947,6.309184,-1.732003,7.331991,-4.046357,8.276047,-2.512927,5.355620,-1.312936,-2.811134,-2.554224,-7.497144,-0.087403,4.327005,4.283357,-1.718452,3.068762,-9.878124,4.981063,-1.537751,7.704060,6.991605,8.385760,9.437609,8.891857,-4.661353,-8.059791,-6.912161,-8.247516,-0.468580,1.244486,-3.151044,-9.897967,-1.246588,-0.494059,0.260198,7.372556,9.272135,2.712147,-9.261470,-8.171524,-8.441526,3.580166,-2.591343,-1.202646,-5.434875,5.369909,-8.588716,4.542806,1.885240,6.410283,8.767879,-6.299888,-8.801344,1.909138,-8.721111,4.803689,-2.373910,3.808843,5.221094,4.286710,3.678078,-4.912665,-7.746629,4.748184,9.504491,4.513033,5.184575,-8.281011,4.568021,-5.419963,2.507740,5.281016,-0.665526,-0.991263,-9.355834,4.790080,-5.594813,-2.074150,6.260085,3.283400,-1.584441,3.956992,-8.866866,5.406655,6.268127,6.883498,-9.726397,-0.214665,4.935215,-3.040575,-2.828394,-9.365299,0.649464,-7.236929,-6.396485,9.955519,-7.594217,-3.932962,-9.036571,-2.032723,4.154679,6.284860,-4.982994,-4.856194,-8.469043,-0.226257,5.799961,-6.630989,3.390954,0.394583,-1.902114,-2.779162,-6.770081,9.342874,9.262852,-4.098300,8.039282,-0.211835,-2.615841,2.620874,6.377945,3.389267,-0.937316,0.619197,-3.134925,0.141329,7.888821,3.902299,-4.611793,3.033713,0.151845,-7.498734,4.928990,-5.127210,-7.841507,-0.018302,7.238071,-9.365577,3.517445,5.460212,-2.630530,-7.343545,2.860500,6.323856,-6.152803,9.670850,4.630164,-7.651556,-2.339039,-5.777143,6.408181,-8.873730,-7.735865,-7.019629,-9.675324,-4.996793,3.119743,-7.428172,-9.228060,-1.736692,-2.450259,-0.211073,-5.741944,-2.400206,-1.742952,7.066917,-5.413881,-0.509585,-8.603497,-1.622422,1.718392,8.126415,-2.275394,5.672175,-2.039516,1.025246,0.027058,-2.980718,2.871831,-1.378377,4.726752,1.329053,6.942498,-4.008486,-2.823300,-9.119045,1.603551,-3.102005,6.458667,3.001066,-3.300325,6.239847,-7.571660,4.597994,-1.416835,3.390199,6.504007,4.898373,0.797325,-1.967718,9.691164,2.177606,-5.548949,-3.815852,2.616887,0.456028,-7.610134,0.786301,-6.648528,-0.811770,-2.513696,-5.118095,-6.629644,-0.036438,-2.500214,-4.080579,6.926677,-4.455395,9.510168,8.357911,-3.109761], dtype = "float64")#candidate|8443|(1280,)|const|float64
call_8442 = relay.TupleGetItem(func_7442_call(relay.reshape(const_8443.astype('float64'), [8, 10, 16])), 0)
call_8444 = relay.TupleGetItem(func_7445_call(relay.reshape(const_8443.astype('float64'), [8, 10, 16])), 0)
func_7480_call = mod.get_global_var('func_7480')
func_7481_call = mutated_mod.get_global_var('func_7481')
call_8450 = relay.TupleGetItem(func_7480_call(), 2)
call_8451 = relay.TupleGetItem(func_7481_call(), 2)
func_5166_call = mod.get_global_var('func_5166')
func_5168_call = mutated_mod.get_global_var('func_5168')
var_8501 = relay.var("var_8501", dtype = "bool", shape = (945,))#candidate|8501|(945,)|var|bool
call_8500 = relay.TupleGetItem(func_5166_call(relay.reshape(var_8501.astype('bool'), [15, 7, 9])), 0)
call_8502 = relay.TupleGetItem(func_5168_call(relay.reshape(var_8501.astype('bool'), [15, 7, 9])), 0)
uop_8503 = relay.log2(var_8501.astype('float32')) # shape=(945,)
output = relay.Tuple([call_8440,call_8442,const_8443,call_8450,call_8500,uop_8503,])
output2 = relay.Tuple([call_8441,call_8444,const_8443,call_8451,call_8502,uop_8503,])
func_8511 = relay.Function([var_8501,], output)
mod['func_8511'] = func_8511
mod = relay.transform.InferType()(mod)
mutated_mod['func_8511'] = func_8511
mutated_mod = relay.transform.InferType()(mutated_mod)
var_8512 = relay.var("var_8512", dtype = "bool", shape = (945,))#candidate|8512|(945,)|var|bool
func_8511_call = mutated_mod.get_global_var('func_8511')
call_8513 = func_8511_call(var_8512)
output = call_8513
func_8514 = relay.Function([var_8512], output)
mutated_mod['func_8514'] = func_8514
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8388_call = mod.get_global_var('func_8388')
func_8390_call = mutated_mod.get_global_var('func_8390')
call_8534 = relay.TupleGetItem(func_8388_call(), 1)
call_8535 = relay.TupleGetItem(func_8390_call(), 1)
func_4889_call = mod.get_global_var('func_4889')
func_4891_call = mutated_mod.get_global_var('func_4891')
call_8543 = func_4889_call()
call_8544 = func_4889_call()
func_8058_call = mod.get_global_var('func_8058')
func_8059_call = mutated_mod.get_global_var('func_8059')
call_8547 = func_8058_call()
call_8548 = func_8058_call()
output = relay.Tuple([call_8534,call_8543,call_8547,])
output2 = relay.Tuple([call_8535,call_8544,call_8548,])
func_8578 = relay.Function([], output)
mod['func_8578'] = func_8578
mod = relay.transform.InferType()(mod)
output = func_8578()
func_8579 = relay.Function([], output)
mutated_mod['func_8579'] = func_8579
mutated_mod = relay.transform.InferType()(mutated_mod)
var_8586 = relay.var("var_8586", dtype = "float32", shape = (15, 11, 2))#candidate|8586|(15, 11, 2)|var|float32
uop_8587 = relay.cos(var_8586.astype('float32')) # shape=(15, 11, 2)
output = relay.Tuple([uop_8587,])
output2 = relay.Tuple([uop_8587,])
func_8589 = relay.Function([var_8586,], output)
mod['func_8589'] = func_8589
mod = relay.transform.InferType()(mod)
var_8590 = relay.var("var_8590", dtype = "float32", shape = (15, 11, 2))#candidate|8590|(15, 11, 2)|var|float32
output = func_8589(var_8590)
func_8591 = relay.Function([var_8590], output)
mutated_mod['func_8591'] = func_8591
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5968_call = mod.get_global_var('func_5968')
func_5970_call = mutated_mod.get_global_var('func_5970')
call_8608 = func_5968_call()
call_8609 = func_5968_call()
func_5200_call = mod.get_global_var('func_5200')
func_5201_call = mutated_mod.get_global_var('func_5201')
call_8621 = relay.TupleGetItem(func_5200_call(), 0)
call_8622 = relay.TupleGetItem(func_5201_call(), 0)
output = relay.Tuple([call_8608,call_8621,])
output2 = relay.Tuple([call_8609,call_8622,])
func_8623 = relay.Function([], output)
mod['func_8623'] = func_8623
mod = relay.transform.InferType()(mod)
mutated_mod['func_8623'] = func_8623
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8623_call = mutated_mod.get_global_var('func_8623')
call_8624 = func_8623_call()
output = call_8624
func_8625 = relay.Function([], output)
mutated_mod['func_8625'] = func_8625
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5968_call = mod.get_global_var('func_5968')
func_5970_call = mutated_mod.get_global_var('func_5970')
call_8643 = func_5968_call()
call_8644 = func_5968_call()
func_1364_call = mod.get_global_var('func_1364')
func_1368_call = mutated_mod.get_global_var('func_1368')
var_8659 = relay.var("var_8659", dtype = "bool", shape = (720,))#candidate|8659|(720,)|var|bool
var_8660 = relay.var("var_8660", dtype = "float64", shape = (1280,))#candidate|8660|(1280,)|var|float64
call_8658 = relay.TupleGetItem(func_1364_call(relay.reshape(var_8659.astype('bool'), [10, 8, 9]), relay.reshape(var_8659.astype('bool'), [10, 8, 9]), relay.reshape(var_8660.astype('float64'), [1280,]), ), 2)
call_8661 = relay.TupleGetItem(func_1368_call(relay.reshape(var_8659.astype('bool'), [10, 8, 9]), relay.reshape(var_8659.astype('bool'), [10, 8, 9]), relay.reshape(var_8660.astype('float64'), [1280,]), ), 2)
func_8578_call = mod.get_global_var('func_8578')
func_8579_call = mutated_mod.get_global_var('func_8579')
call_8667 = relay.TupleGetItem(func_8578_call(), 0)
call_8668 = relay.TupleGetItem(func_8579_call(), 0)
output = relay.Tuple([call_8643,call_8658,var_8659,var_8660,call_8667,])
output2 = relay.Tuple([call_8644,call_8661,var_8659,var_8660,call_8668,])
func_8684 = relay.Function([var_8659,var_8660,], output)
mod['func_8684'] = func_8684
mod = relay.transform.InferType()(mod)
var_8685 = relay.var("var_8685", dtype = "bool", shape = (720,))#candidate|8685|(720,)|var|bool
var_8686 = relay.var("var_8686", dtype = "float64", shape = (1280,))#candidate|8686|(1280,)|var|float64
output = func_8684(var_8685,var_8686,)
func_8687 = relay.Function([var_8685,var_8686,], output)
mutated_mod['func_8687'] = func_8687
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8578_call = mod.get_global_var('func_8578')
func_8579_call = mutated_mod.get_global_var('func_8579')
call_8742 = relay.TupleGetItem(func_8578_call(), 1)
call_8743 = relay.TupleGetItem(func_8579_call(), 1)
output = call_8742
output2 = call_8743
func_8752 = relay.Function([], output)
mod['func_8752'] = func_8752
mod = relay.transform.InferType()(mod)
output = func_8752()
func_8753 = relay.Function([], output)
mutated_mod['func_8753'] = func_8753
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8578_call = mod.get_global_var('func_8578')
func_8579_call = mutated_mod.get_global_var('func_8579')
call_8763 = relay.TupleGetItem(func_8578_call(), 2)
call_8764 = relay.TupleGetItem(func_8579_call(), 2)
func_6666_call = mod.get_global_var('func_6666')
func_6668_call = mutated_mod.get_global_var('func_6668')
call_8787 = relay.TupleGetItem(func_6666_call(), 0)
call_8788 = relay.TupleGetItem(func_6668_call(), 0)
output = relay.Tuple([call_8763,call_8787,])
output2 = relay.Tuple([call_8764,call_8788,])
func_8789 = relay.Function([], output)
mod['func_8789'] = func_8789
mod = relay.transform.InferType()(mod)
mutated_mod['func_8789'] = func_8789
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8789_call = mutated_mod.get_global_var('func_8789')
call_8790 = func_8789_call()
output = call_8790
func_8791 = relay.Function([], output)
mutated_mod['func_8791'] = func_8791
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6816_call = mod.get_global_var('func_6816')
func_6817_call = mutated_mod.get_global_var('func_6817')
call_8813 = func_6816_call()
call_8814 = func_6816_call()
output = call_8813
output2 = call_8814
func_8815 = relay.Function([], output)
mod['func_8815'] = func_8815
mod = relay.transform.InferType()(mod)
output = func_8815()
func_8816 = relay.Function([], output)
mutated_mod['func_8816'] = func_8816
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6517_call = mod.get_global_var('func_6517')
func_6519_call = mutated_mod.get_global_var('func_6519')
call_8838 = relay.TupleGetItem(func_6517_call(), 0)
call_8839 = relay.TupleGetItem(func_6519_call(), 0)
func_6366_call = mod.get_global_var('func_6366')
func_6368_call = mutated_mod.get_global_var('func_6368')
call_8852 = relay.TupleGetItem(func_6366_call(), 0)
call_8853 = relay.TupleGetItem(func_6368_call(), 0)
output = relay.Tuple([call_8838,call_8852,])
output2 = relay.Tuple([call_8839,call_8853,])
func_8866 = relay.Function([], output)
mod['func_8866'] = func_8866
mod = relay.transform.InferType()(mod)
mutated_mod['func_8866'] = func_8866
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8866_call = mutated_mod.get_global_var('func_8866')
call_8867 = func_8866_call()
output = call_8867
func_8868 = relay.Function([], output)
mutated_mod['func_8868'] = func_8868
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6666_call = mod.get_global_var('func_6666')
func_6668_call = mutated_mod.get_global_var('func_6668')
call_8900 = relay.TupleGetItem(func_6666_call(), 0)
call_8901 = relay.TupleGetItem(func_6668_call(), 0)
output = call_8900
output2 = call_8901
func_8918 = relay.Function([], output)
mod['func_8918'] = func_8918
mod = relay.transform.InferType()(mod)
mutated_mod['func_8918'] = func_8918
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8918_call = mutated_mod.get_global_var('func_8918')
call_8919 = func_8918_call()
output = call_8919
func_8920 = relay.Function([], output)
mutated_mod['func_8920'] = func_8920
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6880_call = mod.get_global_var('func_6880')
func_6881_call = mutated_mod.get_global_var('func_6881')
call_8948 = relay.TupleGetItem(func_6880_call(), 0)
call_8949 = relay.TupleGetItem(func_6881_call(), 0)
func_1364_call = mod.get_global_var('func_1364')
func_1368_call = mutated_mod.get_global_var('func_1368')
var_8952 = relay.var("var_8952", dtype = "bool", shape = (720,))#candidate|8952|(720,)|var|bool
const_8953 = relay.const([-7.610455,-5.198907,-2.018558,-9.736469,-2.316960,-1.126868,3.367038,-4.543467,3.379463,4.389511,-9.711380,-8.235865,0.346087,8.785295,-6.296819,-4.318345,-5.418767,-1.457701,-1.892286,-6.506240,-9.684950,-2.460562,-1.810501,-4.553289,-0.840407,-7.966812,8.488033,9.878709,6.275348,6.428231,1.338603,-0.095215,6.787041,-4.570801,1.826232,-8.217150,-5.612828,0.886763,-3.353356,-6.291568,-3.403736,4.637233,-1.168995,-2.194942,1.792051,5.009541,5.238081,-3.243415,-2.740114,-9.358552,1.295311,-8.616386,-7.730451,-0.648145,0.906122,2.470185,7.457180,-1.243137,-2.898590,4.784261,-7.586766,-4.476423,-2.881898,-1.906971,-4.612293,0.923497,-8.149128,1.059293,1.683960,8.521784,-1.025660,-4.701639,8.219154,5.174063,0.652755,5.689330,5.086991,-7.176242,-1.566800,2.543021,9.937083,-8.369074,-6.224273,-8.346992,-0.453704,-4.058333,9.656398,-3.255219,4.140662,-2.911490,8.698408,9.408121,8.680537,-5.156258,-9.618962,-2.199722,-2.443437,6.394593,-9.845034,9.524542,9.523359,-1.749219,-8.959928,6.873240,-2.043892,2.791976,1.136151,7.548806,-7.479146,-0.891127,0.075484,-4.897982,-6.522565,-0.153396,-4.217751,0.548850,8.950192,-6.247634,-8.496228,-6.757737,3.772995,-9.517547,-7.275612,7.818012,-6.148179,1.087167,0.008685,1.470644,-1.101112,-8.281867,0.292292,-8.363435,8.472756,6.497191,9.017985,1.174657,6.524334,0.405169,4.997866,-2.781263,6.224471,4.820352,0.342504,-6.838432,-5.025333,-4.402249,-8.842500,-1.091942,4.454744,3.261925,6.727745,2.198759,7.984745,-3.914153,-2.783896,-0.583677,4.927220,-4.389271,-6.942739,-1.236352,3.868660,9.416504,7.362240,-7.525329,0.066204,5.294789,-8.079172,2.320693,-7.323602,-3.330221,-8.914209,-4.962663,-1.736699,-2.384740,4.934025,9.372352,-4.907282,-2.467776,-6.531827,-8.559161,-0.636030,5.378755,5.883469,-2.275993,-4.291285,-8.986897,7.171868,-7.439948,4.208903,8.347647,3.986480,7.986634,1.360432,-8.854260,-4.789446,9.111789,-7.732177,3.877350,5.145838,7.635266,-7.859068,-2.743016,-8.805541,4.619054,8.705984,-9.628613,3.983628,-6.534200,1.213540,8.786827,-0.557170,5.826816,1.050224,-9.532405,-4.163696,-9.757163,6.944626,6.517797,-1.918479,2.594324,7.173245,-2.073495,9.696142,2.394033,4.730817,3.822030,9.740550,7.683281,9.234262,1.590773,5.943268,1.912228,-9.825152,4.266728,7.866357,-2.488766,-5.351238,-5.798352,-9.686218,8.926631,-7.638308,1.290418,7.515387,-2.394487,-8.409622,-2.594500,0.772723,-7.241810,-6.261263,-4.379604,-3.819336,-2.758607,-6.309201,-8.178985,-4.869641,-2.024128,-2.127066,-0.736273,-5.008302,5.075592,-6.361989,8.588076,0.847944,0.357436,0.414305,2.823736,6.015622,-6.858212,-9.650445,-0.080846,-8.750422,9.898329,-7.510167,-0.841053,-9.091152,1.549085,-6.520561,6.101225,-4.016324,7.807101,-2.810801,6.862251,0.087020,1.410527,-2.793676,-4.248228,4.004830,-6.926522,-6.904875,-4.303584,-8.969757,2.041985,-9.134391,-3.924849,-7.111990,-6.687655,-5.259138,-3.665051,-0.975121,-2.257363,7.606338,0.744122,-8.215627,-1.610387,3.889033,-1.239055,-4.222821,-3.616744,-7.316808,8.290848,-4.029129,7.430719,5.514383,-9.141988,-3.294644,-6.442841,-1.787657,-0.493516,4.029694,2.148211,3.215993,-6.350257,1.099395,8.694352,9.307181,-7.305105,-9.078071,8.983762,9.962348,1.990956,-5.097861,4.929697,-6.900888,4.262306,-6.917094,-9.321350,-0.360022,1.343188,9.313089,0.088517,0.320181,-3.420326,-2.470549,5.796586,-1.723640,3.429212,4.156249,2.136277,-1.115988,4.532317,4.915039,-9.856417,1.946340,9.671760,-3.204557,6.251460,5.892904,-2.990776,8.054008,2.214417,-6.441922,-7.652747,-5.138304,2.124545,1.846774,2.343170,-6.635302,-1.870793,-9.387808,-5.854636,0.748772,-8.735384,-6.739985,-1.576635,3.673917,4.965656,-8.550465,4.993535,2.091125,7.586299,1.447718,0.755199,-8.033514,3.944703,9.436549,-0.048445,-8.039817,0.287956,4.320874,-5.537003,-2.028425,-5.335340,-8.038874,7.420059,0.958375,-5.078870,9.787299,-6.751233,2.735531,-0.919182,-3.727975,6.859091,-2.098622,3.751455,-1.745210,-8.839632,5.187302,3.626538,-8.362238,8.709932,5.028147,-5.959114,4.162876,3.935339,7.462154,2.398716,-0.483148,-0.499525,-0.536299,-7.370374,8.741537,1.275649,9.097019,-2.925217,2.543865,1.041797,1.730410,3.523397,3.099052,-1.413545,6.235015,-8.992372,-9.782540,-9.305731,-1.539777,-4.078117,6.764580,-1.556889,-4.948350,-6.877634,-6.551104,4.331166,4.153563,-8.307871,4.980831,5.094083,-7.238699,-6.532942,-1.992939,6.901838,2.577708,0.567274,6.478182,5.495716,-6.553868,6.312591,-3.290431,4.979201,-2.157774,-5.423046,-5.062476,-8.249909,9.939552,4.479320,7.161129,-9.381357,-8.240959,8.835156,9.694730,-2.002837,-5.944191,-9.232010,-4.802396,-8.129520,9.100223,-0.990144,7.891017,-2.964133,4.055936,-1.278538,3.880359,3.115955,-1.953956,2.893670,0.392043,-8.358095,-6.705218,-9.540356,-5.517623,0.431264,7.520157,1.130983,-8.315005,8.246820,5.855787,-6.080412,-5.872234,9.595799,-1.892859,-7.230825,-5.266516,8.791235,8.410972,3.178482,-1.752652,3.458832,-9.208659,-1.903411,5.687866,-4.619917,-4.871921,6.248962,-9.045341,9.400587,6.982747,-4.342769,-8.720701,-4.843905,-1.961433,0.789650,-0.840546,-4.917029,8.040409,2.328733,5.620181,-3.437034,-4.244068,-6.585645,-5.656124,8.008926,-3.825694,1.953409,0.810544,-3.078760,1.862648,7.733255,0.341286,-0.296298,-7.680020,8.459748,0.642320,4.724541,6.355193,-2.727615,-3.371506,7.074529,1.867889,0.911754,1.761173,-9.861543,-7.673397,-1.621926,-1.193787,-0.451513,5.575010,-6.182693,1.356610,-2.241485,6.943662,-8.394179,5.100628,-6.855753,-7.483080,5.135165,-7.952878,-4.289364,1.679555,-0.101708,8.394667,3.639890,-6.018394,0.228600,3.485685,-7.185586,-5.156654,-9.463232,-0.671409,4.374751,1.584895,-4.777268,-8.660726,-6.214032,-1.063895,8.167546,2.824298,-8.383063,-8.107591,-8.388031,8.812572,9.551391,1.734744,5.008298,-0.322410,6.060982,-2.490140,4.189858,-4.315027,-8.890411,-6.125101,9.618368,-3.851799,9.744880,5.282812,7.339961,-4.532917,2.022752,-7.420028,-6.068635,7.372286,-9.628346,9.541042,-1.941373,9.699277,-1.903064,9.934114,-6.766584,2.474831,-4.348303,1.424189,-1.802589,-5.819459,2.931230,-0.744259,-1.657597,2.652619,8.685893,-9.015093,9.668195,7.735178,-1.370513,7.526163,-6.614204,-8.425281,6.202367,-9.110451,-6.256917,9.744788,-1.219532,8.208263,4.576907,3.840931,0.910650,0.209761,-9.798058,8.082526,9.056690,3.552055,3.348822,-8.902267,9.288815,-1.009904,-2.039292,0.768968,-7.514909,2.310747,-3.038575,-5.626191,-1.797122,-6.397768,-9.889912,5.365654,-5.298259,-3.021953,5.750920,-0.203000,9.666736,-5.198016,8.253686,-6.941755,-6.503439,0.270877,8.703679,-9.938096,9.743273,-4.709839,-1.291782,9.502424,-1.760085,1.861106,2.323889,-1.332764,-5.788015,-4.113697,-1.040155,9.548238,-1.718832,6.390665,8.782383,7.337495,-5.444053,-9.018866,-7.292904,0.167713,2.823597,-8.556532,8.309163,-4.766382,4.538344,3.052816,8.169455,-7.126571,1.643867,3.218326,2.185645,-8.630872,-3.311827,5.555033,-3.285478,8.320660,4.489283,-5.884905,-9.900461,-9.972009,2.481143,-6.865889,9.992328,-6.535986,-8.673542,4.346584,3.423458,-9.288658,-2.013058,-4.922179,-6.294207,8.211716,5.411434,-2.911020,-4.028148,1.286125,-8.556949,-9.813972,-6.664403,3.181188,2.488851,-0.318689,-2.636958,-7.545554,-8.537683,6.839285,3.724219,-3.737756,9.433645,-5.294269,-9.343193,0.948470,5.106133,-0.381755,8.221973,5.586190,-1.458032,9.805137,-0.325102,8.656740,-6.728543,1.192957,1.411768,-7.877159,3.562574,-9.216514,-7.390614,-8.804223,-2.793070,9.077870,-0.202851,-6.772823,-6.562988,2.694017,3.901519,-3.521846,-4.964968,1.365541,1.641690,-0.058972,1.590253,-8.239893,-3.523286,5.498259,9.521845,-3.029986,4.131031,-7.870454,-6.697269,-1.354672,9.401278,5.863569,0.097411,8.381470,-5.663158,6.751241,2.027012,0.875424,-0.590877,-9.417507,8.693950,-0.628219,4.435618,6.545406,-2.362557,6.350578,-4.659237,7.549401,-6.069206,-6.513546,-1.794444,-3.335355,-1.680181,-5.718064,-4.542476,-6.957416,0.286138,0.554267,-5.472201,3.767825,-3.400014,-4.553647,2.887652,-0.158370,-9.392549,9.926032,-3.464245,-9.292175,8.744514,-8.788068,2.578692,-1.698162,9.451928,0.097412,-3.420338,-9.623680,-0.534899,0.578228,2.734663,5.955041,-9.767288,-7.019002,-0.494740,7.999548,0.968122,9.757844,0.744063,-1.811842,8.933080,-0.212504,-3.315686,6.780871,7.578470,-3.003251,-1.074635,-9.982330,-9.567518,-1.697124,-4.188823,5.542092,0.486476,-2.027477,-9.449954,2.438351,7.685964,9.196008,-9.444084,-0.064176,1.822424,-7.180822,-7.011135,-9.698455,-4.477775,2.830617,4.863946,-4.088000,8.847143,0.310991,7.934756,-9.446162,-9.666893,-5.370664,-1.238755,-7.789001,-2.399037,4.157523,0.647385,8.366995,-8.822912,-2.771088,7.358768,-7.840621,-2.112867,-7.333560,-0.025020,-2.966452,-3.582003,-6.487139,9.491414,-5.811466,4.708414,-9.971014,-9.893990,-1.151173,-7.168556,-1.788967,-1.215549,-1.250955,-5.044922,-3.099211,9.196590,8.455156,-6.961364,-0.828238,-2.238288,3.139940,1.496021,-3.785993,9.657537,-0.201588,-8.375340,-8.069166,-6.741146,6.055637,9.326885,9.244839,-2.388501,4.518881,-9.860021,-0.833729,-0.930411,3.076334,6.611895,1.091098,-7.968361,-3.198779,-6.621392,7.892075,6.079872,1.862692,3.787081,-4.530474,9.913142,-4.400342,1.899665,-1.977465,-1.140215,0.112416,-7.094811,-3.217391,7.003311,8.006767,4.130605,-9.518730,6.475701,-4.385057,-2.367152,-1.642321,-7.744181,-5.896779,1.277853,0.090582,-1.286396,8.424085,-1.511954,7.465558,-0.608298,3.094844,-4.721978,-4.511581,-8.250369,-7.033673,6.339252,-1.653005,1.596552,-2.344467,-2.246232,-9.398622,-0.150019,6.516022,3.977532,-7.884458,8.410473,-0.204727,1.458319,2.317815,7.999185,8.555821,-3.164816,-5.258585,-7.526417,-1.427008,-3.475363,7.268879,-6.996602,1.229062,8.196575,7.837593,-3.647922,-8.928217,-2.042222,6.157353,3.089397,5.820550,-9.966438,5.395146,-9.936058,6.314480,-9.341537,5.436882,2.969082,4.549139,4.931652,0.958070,-6.963032,2.087417,4.239870,-9.415430,2.197882,-4.210911,6.072297,-4.722004,6.242428,-4.230988,-0.869342,-8.114468,9.389975,9.866455,-4.492819,-3.833786,-8.640525,-4.418948,-6.088302,5.409271,-3.289846,8.900123,0.563270,0.829467,-8.277384,8.657304,2.114063,-0.152977,2.912867,2.795131,0.336223,-5.337619,-2.858985,5.947828,4.003649,-2.437045,0.475484,6.181440,-4.835445,0.356813,-5.349860,5.555924,5.016481,-7.893485,1.043457,-6.904074,8.566041,7.449617,1.004244,-5.781353,6.100272,-7.816662,7.016580,6.549369,-0.341424,-8.332732,0.249006,0.344298,1.229335,8.145076,4.414405,8.191695,-9.485488,-4.779106,-5.730423,-0.002979,-8.408167,-7.459046,-3.699782,9.034713,-2.516457,-3.117965,4.751129,-8.974646,6.466002,-9.221292,-1.482158,-0.343952,-4.760705,-7.980574,8.072939,-8.957329,-0.100274,9.751439,-3.320016,9.607020,-9.197166,1.165433,9.924205,2.554482,7.554386,-1.049772,8.673261,6.658612,5.016431,-3.849146,-9.635131,-4.599733,1.491535,-3.592793,-8.259840,3.306380,2.921981,2.527341,-8.465315,9.884577,-3.350564,6.624273,-9.803250,5.972394,0.153238,2.406709,-1.299078,-8.593381,9.383245,2.060806,-1.273440,1.398120,2.071462,6.681900,8.550999,-3.850394,-7.098756,6.479444,-9.101838,-9.726787,7.914014,8.070975,-0.134417,3.305029,-1.061317,0.026069,6.234566,-3.800095,2.917197,6.096021,4.573173,-3.290008,6.053415,-7.162396,-4.855665,5.327916,-0.541046,0.144993,-9.209301,8.075781,-2.227457,-5.832765,3.658986,-4.369114,-5.882113,8.829482,9.388958,6.662584,1.028649,-3.918963,-3.042577,1.593283,8.474388,1.008221,-7.755828,-5.414664,-7.887841,-7.883023,-2.608536,-6.985477,-1.297043,-6.206451,-8.228315,2.061776,6.587437,1.059158,7.390483,7.790864,-8.267136,2.541657,8.238029,9.848219,-2.729000,7.524907,-2.251942,8.512415,1.704145,0.389849,1.048201,1.166574,-9.316421,-3.986068,-0.980912,-6.567498,2.916706,3.442786,-7.264621,-1.304572,-6.518421,7.885555,1.212477,-7.645788,-8.706295,-6.409194,-4.348589,-6.481597,6.911956,-5.009064,-9.306586,-0.543317,-8.464253,-9.823162,-6.253890,6.807223,-6.795182,2.123034,9.820083,-7.160074,3.199574,3.428011,2.954489,0.756500,0.221697,-0.100600,-0.129568,5.123491,-2.790120,4.127079,-9.055867,5.505649,0.718838,-9.677325,-3.741511,-1.591409,-4.094959,1.691145,-0.844779,-3.312447,5.081086,6.015417,7.342708,-3.932400,7.707932,-2.764261,5.156300,-0.154771,-1.890913,2.884248,-8.456831,8.296054,-7.270720,-1.733585,2.244774,-1.826782,8.588511,-1.727050,-1.130811,-5.768306,2.480369,9.666388,-6.086543,3.052382,-4.705208,-4.187521,5.715231,4.136962,-2.056048,-5.087244,-8.019922,-2.141660,-2.334362,-3.447125,-7.199488,-8.280603,0.091809,-6.220083,-1.588633], dtype = "float64")#candidate|8953|(1280,)|const|float64
call_8951 = relay.TupleGetItem(func_1364_call(relay.reshape(var_8952.astype('bool'), [10, 8, 9]), relay.reshape(var_8952.astype('bool'), [10, 8, 9]), relay.reshape(const_8953.astype('float64'), [1280,]), ), 2)
call_8954 = relay.TupleGetItem(func_1368_call(relay.reshape(var_8952.astype('bool'), [10, 8, 9]), relay.reshape(var_8952.astype('bool'), [10, 8, 9]), relay.reshape(const_8953.astype('float64'), [1280,]), ), 2)
func_5000_call = mod.get_global_var('func_5000')
func_5005_call = mutated_mod.get_global_var('func_5005')
const_8956 = relay.const([-5.217180,9.613087,-0.901168,-2.571112,-8.483882,1.174849,6.947697,-0.176082,3.644793,-9.806771,2.902814,4.474368,2.208199,5.089620,2.914731,3.515882,-3.239777,3.107197,-1.414766,1.668372,-8.088046,8.518014,9.446707,3.018710,-2.225984,-2.480702,5.625874,-8.658695,-3.436413,0.884693,-1.413349,4.249877,7.103221,-2.484904,-1.898267,-7.864543,-6.341032,-2.048900,-0.733894,1.928142,0.623879,-4.720226,9.117341,-5.295894,-8.927192,7.716409,-6.814781,7.558231,9.873877,9.311639,8.863888,4.464015,-4.569828,6.459220,8.673057,3.096419,-9.736760,7.778396,-3.925801,6.115280,7.948452,-5.871515,3.337343,2.830569,9.058686,-6.242977,-8.382948,-4.376255,8.499280,4.042194,-8.457194,-3.580141,-1.579770,-5.157486,9.710471,0.257889,8.980775,-7.776747,4.015401,6.237981,-4.820247,-3.525764,8.358173,4.404226,-8.259483,0.464073,-1.871254,-9.832139,0.661277,7.074197,-5.008090,9.049192,-6.442761,2.806797,1.689610,-0.023841,-2.084569,3.441303,-4.380053,7.527457,8.200694,3.206480,-1.193724,9.361834,3.012812,-8.150834,0.610029,5.123189,4.333800,-2.436761,0.906288,-2.568549,4.047692,-3.826132,6.738825,-2.471098,-0.824094,-3.100459,3.796051,6.979619,-1.505293,5.246838,6.798850,-4.789868,4.436519,-2.276952,4.445845,-2.072506,-2.556941,-5.776154,-7.576948,-0.606093,1.121313,8.566137,-8.439814,0.127401,-8.660808,0.860906,7.619924,-7.207012,-8.523875,0.241816,-7.987230,-9.695927,-4.119772,-6.878602,6.803005,-1.574810,6.453148,-7.331507,-8.644390,-7.746347,-9.181882,-2.799358,7.040497,-6.620352,0.565835,-2.592210,1.098106,-9.003952,5.952452,9.491258,-0.415855,2.737263,0.999987,1.340409,7.638438,5.164941,4.355397,9.342645,-5.787416,-8.871587,-1.555156,3.711203,5.720186,8.549830,-2.383117,-0.064966,-6.190180,-8.970088,6.518760,0.053828,-2.483205,-8.912914,9.661329,-1.673901,8.189317,0.700635,0.281116,6.309767,6.246617,-7.177829,1.183623,-8.105987,-8.906287,-0.673132,-3.374511,6.012638,-2.060222,0.183825,9.563149,1.126243,-5.535350,-8.491642,-7.261217,2.615107,5.575255,5.275661,-7.871176,-1.485394,0.701861,-1.018852,8.474357,-9.964708,8.025463,5.532335,4.503024,8.171601,9.548535,-5.504061,9.218682,8.892263,4.829356,-6.980047,-5.455165,-5.166305,7.751401,-7.686834,-4.249818,-3.044160,-2.330351,3.418319,-3.930677,1.522881,-0.700846,1.073709,5.583489,-7.518836,-4.079758,-7.072963,-3.707550,-2.831122,3.825850,-0.436727,1.834478,4.084588,-0.770845,3.889149,9.252322,-1.180576,-9.935366,-0.854909,0.610348,-2.934915,8.058397,-2.834565,-8.372405,5.087866,5.884998,6.179695,-9.468558,-1.978345,0.692819,1.736234,2.687264,1.268570,-6.155681,9.418978,-5.321554,1.761935,3.785170,8.158366,-6.484488,4.544693,-4.146191,-9.022753,-6.852916,8.712979,-5.576574,-5.987989,6.662280,-3.859692,7.152550,3.035551,4.629368,4.591512,-0.449618,-0.115320,7.047413,4.260105,5.431465,-2.975415,-0.096599,4.735102,9.845754,-9.175450,8.171982,-6.446582,5.246703,4.621810,-1.794770,7.357901,8.917293,9.140272,2.173102,8.418884,-1.816521,-5.621975,4.586153,9.463900,-8.290881,8.789378,-6.800533,-2.484995,5.333780,3.506392,8.182001,7.867153,-2.399661,5.713764,9.925613,-6.625690,6.668315,-3.957249,0.556974,0.933734,-4.854579,0.685920,-6.594594,-5.017881,9.748020,6.194431,-2.599590,-4.108819,9.346163,-4.815502,-9.819087,-8.737059,-1.454184,-9.874640,-3.495430,6.878883,7.457856,-6.530470,-7.794141,4.938069,-9.407097,-9.007055,3.506214,9.902026,2.101669,4.028511,9.892727,6.096867,9.231394,2.054880,3.619813,-0.171619,0.530840,-8.152785,4.195175,-5.538434,-4.588064,-7.691921,5.331487,-8.502133,4.466110,-7.396524,1.552953,-7.931477,8.684267,-0.782582,-4.350194,5.266238,1.409492,-4.305672,-0.817141,-9.244392,3.669034,-4.139389,7.422209,5.843204,1.124621,9.283522,7.765465,7.754745,4.087441,8.776039,-4.160787,0.666750,-8.775549,-2.809131,-3.070111,1.666890,7.619588,-8.447793,-0.086830,1.450912,4.294852,-8.935377,1.175201,-8.284887,-4.751968,-5.967225,-2.890020,-8.243775,8.605712,9.823280,-2.932088,-3.004943,7.049146,-1.904121,-0.273124,4.534517,-7.486883,9.086017,5.013883,-7.392502,8.881408,-8.479490,-2.222396,9.734913,7.841497,-4.217003,8.245907,3.822539,9.448334,4.983633,4.250379,-8.365113,-3.946051,2.300227,9.099155,9.729448,9.020970,-6.422030,0.512281,5.746515,2.658467,-1.009626,-9.557110,4.917062,1.369782,-0.974670,1.695737,-5.617329,-2.081250,7.197231,1.846843,-9.097640,-5.984949,-9.882926,-3.585322,-6.232489,6.666845,6.518170,-5.555923,-1.632569,6.751111,-3.389710,6.345496,-7.896489,7.913372,7.173032,0.008563,3.488145,-6.747139,1.118947,0.699987,-5.567013,-9.991141,5.262792,-4.695042,-7.619769,-9.703112,-2.911436,1.256109,6.153489,-3.634190,1.633311,-0.433437,6.454907,-3.879787,-9.100175,-8.488454,8.071642,-3.924525,9.511125,-8.333264,-5.857175,3.485516,-5.762224,9.468334,7.034276,-8.315890,9.040102,4.542019,5.794826,5.364565,9.315516,3.517728,-9.737177,1.875786,-2.549592,9.897497,-0.742732,-8.199432,2.174515,-3.413525,4.649175,7.241282,-4.037611,-3.669032,4.119361,3.470466,8.182623,4.750265,-7.068145,-7.847124,-1.130475,0.866113,-0.018964,7.820458,1.103798,-5.363129,-2.818983,-4.076229,-2.605762,-9.761796,-5.539449,-8.788914,-9.732837,-2.304722,4.543176,-3.233081,-0.360591,-2.886346,9.975145,8.378556,-1.404426,7.243803,5.235047,1.303099,-6.560328,7.997912,7.106452,3.277827,1.656797,-9.259432,-9.775822,-3.266486,-2.951902,6.346094,-6.284260,-4.357175,-2.502613,6.627995,6.064590,6.592654,3.871458,6.062214,-5.947727,-1.082754,-1.492901,-1.414071,3.938368,1.991621,-2.367455,7.582633,2.113649,3.196028,-9.250741,-2.358759,9.131038,2.573886,-2.997369,6.301069,5.049082,1.652829,-3.996963,-0.773789,-1.246054,8.999035,-4.301192,-4.942133,-2.659149,1.954773,-4.282321,9.786402,8.561647,-9.907371,4.609186,8.322299,-6.384616,-9.927895,-4.828371,-0.417468,-0.966635,6.369747,6.719659,7.137751,2.471824,-6.502306,-2.715884,2.034969,-0.846450,4.564579,7.848382,9.844528,2.913017,-5.305989,-9.147604,1.607079,-8.925062,-2.118398,-1.308517,-7.336024,-0.065913,5.921340,-3.495179,7.812721,0.959517,-0.440304,8.478646,-1.717374,4.072296,-3.282730,6.310914,-8.126253,5.207473,7.018876,-6.046092,5.537674,-0.817967,2.848814,-6.168566,-2.507886,-3.019939,-7.670319,-4.390539,-5.863211,-0.589579,4.664438,-5.479092,7.204082,-7.267594,3.019989,-0.485326,-6.501616,-4.359902,1.741936,-7.513841,-5.004321,-2.145063,-1.246490,5.411296,-2.625897,9.149109,0.772831,-6.550400,6.482053,-7.931976,2.561928,-9.081176,2.477708,-1.718755,3.495101,-7.134898,5.891101,-2.557757,8.253616,-7.106383,7.511112,-6.896552,-2.565901,-6.356371,5.278815,-9.149193,9.783837,-1.173064,5.440052,-1.555295,0.090055,-7.177643,0.367440,-8.020864,-5.143775,1.578620,7.530053,8.560398,-9.731928,5.777388,-3.829950,-0.713864,-0.526853,-9.107535,3.525556,-3.961902,-8.349875,6.012461,-4.844894,-8.087726,4.855789,-8.963459,1.988422,-4.224043,9.156471,-8.942183,8.347817,-8.983645,5.953908,-0.185040,7.144014,-2.269784,6.932105,6.242328,7.284004,4.047974,1.218579,9.495000,4.921497,-9.315766,-8.831353,-7.178722,-9.281873,4.748859,6.744739,-7.777808,1.089938,-2.542115,1.529083,-1.381958,-2.262894,8.430025,6.482099,-3.605318,-8.343537,-9.445430,0.467149,-8.371279,2.517799,-9.736277,2.444225,4.590296,1.928479,-5.397029,2.745957,8.938931,-3.114557,-3.665615,-2.379640,-3.446348,9.508951,4.958893,9.404184,-3.289041,3.044537,9.883414,-5.687363,0.101132,0.040639,-5.868068,-7.699296,-4.057460,-9.769694,2.210581,5.611166,6.074638,-0.957391,-7.251585,-9.007401,-5.648916,7.378779,7.864350,-1.917211,5.783042,-5.269934,-9.901262,9.362225,7.056719,6.950552,-8.831439,-0.219453,-5.383322,-2.621088,-3.375838,6.069032,6.771281,-0.150474,-9.636020,9.483476,-8.631894], dtype = "float32")#candidate|8956|(792,)|const|float32
const_8957 = relay.const([-1,-1,-6,-1,9,-10,7,5,-9,-9,-3,9,-2,1,4,-9,5,-3,8,-9,-7,3,6,-8,-8,-3,1,7,1,-9,5,7,-7,-5,8,1,3,-9,-3,4,-3,-2,-3,10,-8,8,-6,-7,9,7,4,-1,3,5,-9,5,8,-8,8,-4,10,4,-8,8,-3,-5,4,-3,7,1,9,-10,5,9,-8,4,6,4,8,-3,1,-7,5,-2,-10,-10,10,7,10,1,-10,7,-2,9,-5,1,2,10,10,2,6,-8,-4,2,-2,3,9,-5,1,10,-10,-3,-9,-2,7,-8,9,-9,-2,6,-7,5,9,4,-7,7,5,2,2,-2,9,4,-7,-1,-6,-4,-8,-9,-10,6,-9,-7,-2,8,-10,-3,-10,6,-9,-6,-8,10,-1,2,1,-1,10,5,-4,5,3,4,6,3,-1,-10,1,-2,8,4,-3,-8,-10,4,-5,-7,-9,-4,-2,-10,-5,-9,-2,-7,-6,-10,5,-9,-10,-6,-4,-5,-5,-9,3,-10,-2,5,2,4,4,-5,1,-3,-1,-8,-3,9,2,-4,7,-5,5,-3,-5,-2,-1,7,3,3,-8,-8,-3,-8,-6,2,-8,-4,-9,9,6,-1,8,-6,-8,4,1,-7,-8,-1,-9,8,10,-10,1,-10,-1,-1,-1,-2,10,-4,-8,7,-6,-2,9,-2,-4,-7,4,-9,8,-3,-8,5,1,-9,10,6,-2,1,-10,2,7,-8,-5,-5,-8,-2,-5,4,-6,-4,-9,-4,-1,-4,-10,-5,-6,-10,-2,5,-3,-6,-4,4,-3,8,-9,3,-7,-10,-3,-6,8,1,-8,9,1,-2,6,-7,-2,-1,7,-2,-10,3,-9,-2,4,-8,10,-6,3,-8,-3,7,9,-8,-3,5,5,7], dtype = "uint16")#candidate|8957|(336,)|const|uint16
const_8958 = relay.const([-3,-4,7,-6,-5,-2,7,-9,3,-4,8,-3,3,1,4,-10,-4,7,10,-8,-2,9,8,-6,8,1,-9,7,4,3,1,-6,10,8,-7,-5,1,-8,2,9,8,-5,6,-1,1,10,-7,-7,-10,-7,5,3,5,-2,-9,-6,3,-4,-5,-9,8,8,-8,2,-4,-2,-10,3,-10,2,-8,-6,2,-9,-4,-5,10,8,2,5,-1,-5,3,6,-4,5,-9,-2,1,-1,-2,7,-10,9,9,8,-2,-8,-10,8,-5,6,4,-6,-1,-9,-3,-4,-3,5,-8,1,7,-3,-6,-10,6,4,6,-4,-8,8,9,-4,-5,-6,5,3,7,-6,-8,5,-4,3,-10,5,-10,-1,6,-5,8,4,-1,3,2,2,-8,6,-5,-4,-8,6,3,-10,5,-10,9,6,10,7,-3,-5,-5,4,2,8,6,-1,-1,8,8,-1,-9,-6,-10,-2,-3,3,8,9,-5,3,1,10,6,-10,-5,1,1,4,2,2,5,-5,-3,4,-1,9,2,4,9,-2,4,-8,-3,-6,-10,-2,4,-7,-2,-6,1,-5,-7,9,-4,8,-6,-2,-7,-4,9,-2,-5,-1,-2,-2,-2,-6,-7,-3,-7,10,7,10,2,-1,-4,-3,3,-8,1,-10,8,-2,-10,-7,-5,-10,-8,-6,-8,1,-3,-1,-6,-3,-10,10,8,4,-6,-7,-7,-3,9,5,-10,9,1,-10,-10,6,8,7,-7,-6,-1,3,-9,-3,10,6,-6,-3,-5,-4,8,5,5,1,-1,4,-9,-10,-7,-7,-4,-7,7,7,-7,-9,10,-10,-5,6,8,-2,9,5,6,3,4,10,1,5,6,-10,9,-3,3,-4,9,9,-7,-4,-6,-3,-6,-4,-8,-3,-7,5,7,2,-7,-2,-2,6,5,4,-6,4,2,9,10,-6,-8,-5,-7,4,5,2,-7,9,-8,-3,-2,8,-6,-3,2,-5,10,-5,7,5,-5,9,3,4,-3,-8,6,8,-7,-10,10,-9,-5,1,4,-4,-1,3,-3,5,-5,-10,1,7,-6,-6,4,9,8,1,-6,-2,-8,-2,1,-4,2,-9,9,-10,-5,-3,10,8,4,1,-4,8,6,-2,9,5,-9,6,-2,-5,3,-6,2,2,-5,7,-5,-3,6,2,-1,-9,-3,10,-1,4,5,-8,6,-4,-4,2,5,-2,9,8,-1,10,-2,-9,7,7,-5,-8,5,-4,2,-8,-1,10,4,-2,7,-9,4,1,3,-3,3,9,-3,6,-9,4,7,-2,4,-6,-7,9,6,3,8,-8,-7,-10,10,-8,2,-10,8,8,7,-10,-2,4,-8,7,-2,8,4,10,6,4,1,-5,-8,-8,10,7,-8,-9,-9,4,-3,-1,-10,2,2,-2,-2,10,3,-4,10,6,-4,6,-1,-7,1,-1,2,7,-1,-6,9,2,-8,6,8,5,4,9,2,-2,-5,-3,8,10,-2,-2,-4,2,1,-7,-9,10,6,1,-5,-10,-7,9,5,-7,-8,8,3,2,-10,-2,-5,-4,-4,-8,9,-1,-1,6,-2,4,1,-9,-6,10,-4,-6,-7,-10,9,-9,-6,-7,-3,-4,5,2,-2,2,-4,5,-5,6,5,-2,-6,3,3,8,2,-8,9,-4,-3,-10,-1,4,8,-9,-8,7,3,-5,-9,-2,-4,-1,4,-6,4,-3,-3,-2,-9,-5,-3,2,-4,4,-9,-8,1,-7,-3,-10,-9,3,-2,-4,-7,9,-5,9,7,5,-2,-10,-10,-9,-10,-1,4,2,8,-6,-6,6,5,1,9,-9,-6,-7,5,-1,-5,-3,-6,-2,10,-9,2,-10,-1,8,-8,2,-5,2,4,-5,8,7,-3,-8,6,1,-5,4,5,-7,-8,-7,-9,-5,4,2,-7,6,8,5,1,4,9,1,10,5,6,-3,4,-8,7], dtype = "int8")#candidate|8958|(729,)|const|int8
call_8955 = relay.TupleGetItem(func_5000_call(relay.reshape(const_8956.astype('float32'), [792,]), relay.reshape(const_8957.astype('uint16'), [336,]), relay.reshape(const_8958.astype('int8'), [729,]), ), 6)
call_8959 = relay.TupleGetItem(func_5005_call(relay.reshape(const_8956.astype('float32'), [792,]), relay.reshape(const_8957.astype('uint16'), [336,]), relay.reshape(const_8958.astype('int8'), [729,]), ), 6)
func_6506_call = mod.get_global_var('func_6506')
func_6510_call = mutated_mod.get_global_var('func_6510')
call_8963 = relay.TupleGetItem(func_6506_call(relay.reshape(call_8951.astype('float64'), [64, 20]), relay.reshape(const_8957.astype('uint16'), [336,]), ), 3)
call_8964 = relay.TupleGetItem(func_6510_call(relay.reshape(call_8951.astype('float64'), [64, 20]), relay.reshape(const_8957.astype('uint16'), [336,]), ), 3)
output = relay.Tuple([call_8948,call_8951,var_8952,const_8953,call_8955,const_8956,const_8957,const_8958,call_8963,])
output2 = relay.Tuple([call_8949,call_8954,var_8952,const_8953,call_8959,const_8956,const_8957,const_8958,call_8964,])
func_8966 = relay.Function([var_8952,], output)
mod['func_8966'] = func_8966
mod = relay.transform.InferType()(mod)
mutated_mod['func_8966'] = func_8966
mutated_mod = relay.transform.InferType()(mutated_mod)
var_8967 = relay.var("var_8967", dtype = "bool", shape = (720,))#candidate|8967|(720,)|var|bool
func_8966_call = mutated_mod.get_global_var('func_8966')
call_8968 = func_8966_call(var_8967)
output = call_8968
func_8969 = relay.Function([var_8967], output)
mutated_mod['func_8969'] = func_8969
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8058_call = mod.get_global_var('func_8058')
func_8059_call = mutated_mod.get_global_var('func_8059')
call_8971 = func_8058_call()
call_8972 = func_8058_call()
func_4185_call = mod.get_global_var('func_4185')
func_4190_call = mutated_mod.get_global_var('func_4190')
var_8976 = relay.var("var_8976", dtype = "float32", shape = (792,))#candidate|8976|(792,)|var|float32
var_8977 = relay.var("var_8977", dtype = "float64", shape = (1280,))#candidate|8977|(1280,)|var|float64
const_8978 = relay.const([7,-8,-7,10,-5,-5,2,-10,3,2,-4,-9,1,-5,1,7,-9,-9,-2,-10,-4,7,-4,-10,-6,2,-7,9,2,5,10,-9,9,-7,-9,-7,-9,-10,-5,3,-5,1,8,8,-6,1,-6,-4,10,-5,-8,5,1,2,8,-5,-2,2,8,10,-4,-1,1,7,-9,-4,7,4,-5,-1,-4,3,-7,-6,7,-1,6,1,-7,3,-1,3,-4,-3,7,-3,2,-9,-3,4,-2,10,6,5,10,-9,5,-8,-3,10,-6,-3,1,5,-8,3,-3,7,-4,3,10,-6,8,-6,6,-2,-8,-1,8,2,-9,5,-6,3,3,-4,10,10,6,10,1,-6,-7,-8,5,1,3,-5,-6,-8,4,6,4,3,-1,1,5,10,4,-6,-8,-4,3,-1,-8,10,1,-8,-1,4,-4,-5,-4,-2,5,-9,7,3,3,-10,2,-2,5,9,-1,-1,2,3,-1,4,2,-8,-7,8,-4,10,8,1,2,-3,-5,-7,10,-2,-4,6,10,-7,-6,6,5,8,-7,-9,10,-9,6,-3,5,-7,2,7,1,10,10,9,-1,1,7,7,3,4,4,-6,-3,-10,-10,8,2,9,5,9,7,5,1,-3,8,4,5,10,4,-6,8,8,-3,4,-4,-5,5,7,2,8,7,3,-3,1,-3,-6,2,-1,-8,4,-9,7,1,-8,1,-4,-10,-2,1,4,2,-5,3,-1,2,-5,6,-5,-5,1,-10,4,-3,-7,-2,8,-1,8,2,5,5,-7,3,8,-4,3,6,-6,-6,3,1,1,6,-6,-7,4,-2,-2,2,2,-1,-10,4,5,-6,-9,-10,1,-8,-8,-9,-8,3,7,-10,-1,-3,7,6,9,-10,3,9,-10], dtype = "uint16")#candidate|8978|(336,)|const|uint16
call_8975 = relay.TupleGetItem(func_4185_call(relay.reshape(var_8976.astype('float32'), [8, 11, 9]), relay.reshape(var_8977.astype('float64'), [1280,]), relay.reshape(const_8978.astype('uint16'), [336,]), ), 2)
call_8979 = relay.TupleGetItem(func_4190_call(relay.reshape(var_8976.astype('float32'), [8, 11, 9]), relay.reshape(var_8977.astype('float64'), [1280,]), relay.reshape(const_8978.astype('uint16'), [336,]), ), 2)
func_5442_call = mod.get_global_var('func_5442')
func_5444_call = mutated_mod.get_global_var('func_5444')
call_8998 = func_5442_call()
call_8999 = func_5442_call()
func_5968_call = mod.get_global_var('func_5968')
func_5970_call = mutated_mod.get_global_var('func_5970')
call_9000 = func_5968_call()
call_9001 = func_5968_call()
output = relay.Tuple([call_8971,call_8975,var_8976,var_8977,const_8978,call_8998,call_9000,])
output2 = relay.Tuple([call_8972,call_8979,var_8976,var_8977,const_8978,call_8999,call_9001,])
func_9004 = relay.Function([var_8976,var_8977,], output)
mod['func_9004'] = func_9004
mod = relay.transform.InferType()(mod)
mutated_mod['func_9004'] = func_9004
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9004_call = mutated_mod.get_global_var('func_9004')
var_9006 = relay.var("var_9006", dtype = "float32", shape = (792,))#candidate|9006|(792,)|var|float32
var_9007 = relay.var("var_9007", dtype = "float64", shape = (1280,))#candidate|9007|(1280,)|var|float64
call_9005 = func_9004_call(var_9006,var_9007,)
output = call_9005
func_9008 = relay.Function([var_9006,var_9007,], output)
mutated_mod['func_9008'] = func_9008
mutated_mod = relay.transform.InferType()(mutated_mod)
var_9012 = relay.var("var_9012", dtype = "float32", shape = (1, 4, 16))#candidate|9012|(1, 4, 16)|var|float32
uop_9013 = relay.atanh(var_9012.astype('float32')) # shape=(1, 4, 16)
func_7613_call = mod.get_global_var('func_7613')
func_7617_call = mutated_mod.get_global_var('func_7617')
var_9020 = relay.var("var_9020", dtype = "uint16", shape = (336,))#candidate|9020|(336,)|var|uint16
var_9021 = relay.var("var_9021", dtype = "float32", shape = (396, 2))#candidate|9021|(396, 2)|var|float32
call_9019 = relay.TupleGetItem(func_7613_call(relay.reshape(var_9020.astype('uint16'), [336, 1]), relay.reshape(var_9021.astype('float32'), [792,]), ), 1)
call_9022 = relay.TupleGetItem(func_7617_call(relay.reshape(var_9020.astype('uint16'), [336, 1]), relay.reshape(var_9021.astype('float32'), [792,]), ), 1)
func_5908_call = mod.get_global_var('func_5908')
func_5912_call = mutated_mod.get_global_var('func_5912')
var_9027 = relay.var("var_9027", dtype = "uint16", shape = (33, 4))#candidate|9027|(33, 4)|var|uint16
call_9026 = relay.TupleGetItem(func_5908_call(relay.reshape(var_9027.astype('uint16'), [11, 12, 1]), relay.reshape(var_9021.astype('uint16'), [11, 12, 6]), ), 1)
call_9028 = relay.TupleGetItem(func_5912_call(relay.reshape(var_9027.astype('uint16'), [11, 12, 1]), relay.reshape(var_9021.astype('uint16'), [11, 12, 6]), ), 1)
output = relay.Tuple([uop_9013,call_9019,var_9020,var_9021,call_9026,var_9027,])
output2 = relay.Tuple([uop_9013,call_9022,var_9020,var_9021,call_9028,var_9027,])
func_9029 = relay.Function([var_9012,var_9020,var_9021,var_9027,], output)
mod['func_9029'] = func_9029
mod = relay.transform.InferType()(mod)
var_9030 = relay.var("var_9030", dtype = "float32", shape = (1, 4, 16))#candidate|9030|(1, 4, 16)|var|float32
var_9031 = relay.var("var_9031", dtype = "uint16", shape = (336,))#candidate|9031|(336,)|var|uint16
var_9032 = relay.var("var_9032", dtype = "float32", shape = (396, 2))#candidate|9032|(396, 2)|var|float32
var_9033 = relay.var("var_9033", dtype = "uint16", shape = (33, 4))#candidate|9033|(33, 4)|var|uint16
output = func_9029(var_9030,var_9031,var_9032,var_9033,)
func_9034 = relay.Function([var_9030,var_9031,var_9032,var_9033,], output)
mutated_mod['func_9034'] = func_9034
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7656_call = mod.get_global_var('func_7656')
func_7657_call = mutated_mod.get_global_var('func_7657')
call_9036 = relay.TupleGetItem(func_7656_call(), 0)
call_9037 = relay.TupleGetItem(func_7657_call(), 0)
output = relay.Tuple([call_9036,])
output2 = relay.Tuple([call_9037,])
func_9045 = relay.Function([], output)
mod['func_9045'] = func_9045
mod = relay.transform.InferType()(mod)
mutated_mod['func_9045'] = func_9045
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9045_call = mutated_mod.get_global_var('func_9045')
call_9046 = func_9045_call()
output = call_9046
func_9047 = relay.Function([], output)
mutated_mod['func_9047'] = func_9047
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5642_call = mod.get_global_var('func_5642')
func_5644_call = mutated_mod.get_global_var('func_5644')
call_9062 = relay.TupleGetItem(func_5642_call(), 2)
call_9063 = relay.TupleGetItem(func_5644_call(), 2)
output = call_9062
output2 = call_9063
func_9072 = relay.Function([], output)
mod['func_9072'] = func_9072
mod = relay.transform.InferType()(mod)
output = func_9072()
func_9073 = relay.Function([], output)
mutated_mod['func_9073'] = func_9073
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8253_call = mod.get_global_var('func_8253')
func_8255_call = mutated_mod.get_global_var('func_8255')
call_9106 = func_8253_call()
call_9107 = func_8253_call()
output = call_9106
output2 = call_9107
func_9127 = relay.Function([], output)
mod['func_9127'] = func_9127
mod = relay.transform.InferType()(mod)
output = func_9127()
func_9128 = relay.Function([], output)
mutated_mod['func_9128'] = func_9128
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6366_call = mod.get_global_var('func_6366')
func_6368_call = mutated_mod.get_global_var('func_6368')
call_9193 = relay.TupleGetItem(func_6366_call(), 0)
call_9194 = relay.TupleGetItem(func_6368_call(), 0)
output = call_9193
output2 = call_9194
func_9202 = relay.Function([], output)
mod['func_9202'] = func_9202
mod = relay.transform.InferType()(mod)
mutated_mod['func_9202'] = func_9202
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9202_call = mutated_mod.get_global_var('func_9202')
call_9203 = func_9202_call()
output = call_9203
func_9204 = relay.Function([], output)
mutated_mod['func_9204'] = func_9204
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9045_call = mod.get_global_var('func_9045')
func_9047_call = mutated_mod.get_global_var('func_9047')
call_9205 = relay.TupleGetItem(func_9045_call(), 0)
call_9206 = relay.TupleGetItem(func_9047_call(), 0)
func_8966_call = mod.get_global_var('func_8966')
func_8969_call = mutated_mod.get_global_var('func_8969')
const_9252 = relay.const([[False],[True],[False],[False],[True],[False],[True],[False],[True],[True],[False],[False],[False],[True],[False],[False],[True],[True],[False],[False],[False],[False],[True],[False],[False],[False],[False],[True],[False],[True],[False],[False],[False],[False],[True],[False],[False],[False],[False],[False],[True],[False],[False],[False],[False],[False],[False],[True],[True],[False],[False],[True],[False],[True],[False],[False],[False],[False],[True],[False],[True],[False],[True],[False],[False],[False],[False],[True],[False],[False],[True],[True],[True],[True],[True],[True],[False],[True],[False],[True],[True],[False],[True],[False],[False],[True],[False],[False],[True],[False],[False],[False],[False],[True],[False],[False],[True],[False],[True],[True],[True],[False],[True],[False],[False],[False],[True],[False],[False],[True],[True],[True],[False],[False],[True],[False],[True],[False],[False],[False],[False],[False],[True],[False],[False],[True],[False],[True],[False],[False],[True],[True],[False],[False],[True],[False],[False],[False],[False],[False],[True],[False],[False],[True],[False],[True],[True],[False],[True],[True],[False],[True],[False],[True],[True],[False],[True],[False],[True],[True],[False],[False],[False],[False],[False],[True],[True],[True],[True],[True],[True],[False],[True],[True],[False],[False],[True],[True],[False],[False],[True],[False],[False],[True],[False],[True],[True],[True],[True],[False],[False],[True],[True],[False],[False],[True],[True],[True],[False],[False],[False],[False],[False],[False],[True],[False],[False],[False],[False],[False],[True],[True],[False],[True],[False],[False],[False],[True],[True],[False],[True],[True],[True],[False],[True],[True],[False],[False],[False],[False],[False],[True],[False],[True],[True],[False],[True],[True],[False],[True],[True],[True],[False],[True],[False],[False],[False],[True],[True],[True],[True],[True],[True],[False],[True],[False],[False],[True],[True],[False],[False],[True],[True],[False],[False],[False],[False],[False],[True],[True],[True],[False],[False],[False],[False],[True],[False],[False],[False],[False],[True],[True],[True],[False],[True],[False],[True],[True],[False],[False],[False],[False],[True],[True],[False],[False],[False],[True],[False],[True],[False],[True],[False],[False],[True],[False],[True],[False],[False],[True],[False],[True],[False],[True],[False],[False],[True],[True],[True],[True],[True],[True],[True],[False],[True],[False],[False],[True],[True],[False],[False],[True],[False],[False],[True],[True],[False],[True],[True],[False],[False],[True],[False],[True],[False],[False],[True],[False],[True],[False],[True],[False],[False],[True],[False],[True],[True],[True],[False],[False],[True],[False],[True],[True],[True],[False],[True],[True],[True],[False],[False],[False],[False],[False],[True],[False],[False],[False],[False],[True],[False],[False],[True],[False],[True],[True],[True],[False],[False],[True],[False],[False],[False],[True],[True],[True],[True],[False],[False],[True],[True],[False],[True],[True],[False],[False],[True],[False],[False],[False],[True],[False],[False],[False],[True],[True],[False],[False],[True],[False],[False],[True],[False],[False],[False],[False],[True],[True],[False],[True],[False],[True],[False],[True],[False],[False],[True],[False],[True],[True],[False],[False],[False],[False],[True],[True],[True],[True],[True],[False],[True],[True],[True],[False],[True],[True],[False],[False],[False],[False],[False],[True],[False],[False],[False],[False],[False],[False],[False],[True],[True],[False],[False],[True],[False],[True],[False],[True],[False],[False],[True],[True],[True],[False],[True],[True],[True],[True],[True],[False],[False],[True],[True],[False],[True],[True],[False],[False],[True],[True],[True],[True],[True],[True],[False],[False],[False],[True],[True],[False],[True],[True],[False],[False],[True],[True],[True],[True],[True],[True],[True],[True],[False],[False],[True],[False],[True],[True],[False],[True],[False],[True],[False],[False],[True],[False],[False],[True],[True],[True],[False],[False],[False],[False],[True],[False],[True],[False],[True],[False],[True],[False],[True],[False],[False],[False],[False],[True],[False],[False],[False],[False],[True],[False],[False],[False],[True],[False],[False],[False],[False],[False],[True],[False],[False],[False],[False],[True],[False],[True],[False],[False],[True],[True],[False],[True],[False],[False],[True],[False],[False],[True],[False],[True],[False],[True],[False],[True],[True],[False],[True],[True],[False],[False],[False],[False],[False],[False],[True],[True],[False],[True],[True],[True],[True],[True],[True],[True],[True],[False],[True],[True],[True],[True],[False],[True],[False],[False],[False],[False],[False],[False],[True],[False],[False],[True],[False],[True],[False],[False],[False],[False],[True],[True],[True],[False],[False],[False],[True],[True],[False],[False],[False],[False],[False],[False],[True],[False],[True],[False],[False],[True],[False],[False],[False],[False],[True],[False],[True],[True],[False],[True],[True],[False],[False],[True],[False],[False],[True],[False],[True],[False],[False],[True],[False],[False],[False],[True],[True],[True],[True],[True],[False],[True],[True],[False],[False],[True],[True],[True],[False],[True],[True],[False],[True],[True],[True],[False],[False],[False],[False],[True],[False],[False],[True],[False],[True],[True],[False],[False]], dtype = "bool")#candidate|9252|(720, 1)|const|bool
call_9251 = relay.TupleGetItem(func_8966_call(relay.reshape(const_9252.astype('bool'), [720,])), 6)
call_9253 = relay.TupleGetItem(func_8969_call(relay.reshape(const_9252.astype('bool'), [720,])), 6)
output = relay.Tuple([call_9205,call_9251,const_9252,])
output2 = relay.Tuple([call_9206,call_9253,const_9252,])
func_9257 = relay.Function([], output)
mod['func_9257'] = func_9257
mod = relay.transform.InferType()(mod)
output = func_9257()
func_9258 = relay.Function([], output)
mutated_mod['func_9258'] = func_9258
mutated_mod = relay.transform.InferType()(mutated_mod)
var_9275 = relay.var("var_9275", dtype = "float32", shape = (12, 6, 6))#candidate|9275|(12, 6, 6)|var|float32
uop_9276 = relay.atan(var_9275.astype('float32')) # shape=(12, 6, 6)
output = relay.Tuple([uop_9276,])
output2 = relay.Tuple([uop_9276,])
func_9279 = relay.Function([var_9275,], output)
mod['func_9279'] = func_9279
mod = relay.transform.InferType()(mod)
mutated_mod['func_9279'] = func_9279
mutated_mod = relay.transform.InferType()(mutated_mod)
var_9280 = relay.var("var_9280", dtype = "float32", shape = (12, 6, 6))#candidate|9280|(12, 6, 6)|var|float32
func_9279_call = mutated_mod.get_global_var('func_9279')
call_9281 = func_9279_call(var_9280)
output = call_9281
func_9282 = relay.Function([var_9280], output)
mutated_mod['func_9282'] = func_9282
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6794_call = mod.get_global_var('func_6794')
func_6795_call = mutated_mod.get_global_var('func_6795')
call_9452 = relay.TupleGetItem(func_6794_call(), 0)
call_9453 = relay.TupleGetItem(func_6795_call(), 0)
func_5572_call = mod.get_global_var('func_5572')
func_5577_call = mutated_mod.get_global_var('func_5577')
var_9455 = relay.var("var_9455", dtype = "int8", shape = (528,))#candidate|9455|(528,)|var|int8
var_9456 = relay.var("var_9456", dtype = "float32", shape = (792,))#candidate|9456|(792,)|var|float32
const_9457 = relay.const([[7,4,-7,-9,-4,-4,-10,-2,-2,9,6,-6,-4,4,3,4,2,-3,3,-7,-7,-1,-5,4,-7,-3,3,-10,8,-7,-7,-4,9,9,-1,6,-7,-3,-10,-10,-1,2,4,-3,5,7,2,-3,9,2,8,6,-7,-10,7,5,-4,-4,-8,-3,8,5,1,1,-7,10,3,3,-8,-6,-4,4,10,6,-4,1,9,4,-1,10,7,7,7,-9,-4,9,-8,4,1,-3,1,9,-9,1,-7,10,-2,-6,-7,-5,10,9,-9,3,4,-6,-2,-4,7,1,-8,-10,7,3,-10,7,6,-7,8,4,-10,5,-5,-2,-6,-6,5,-7,1,8,-10,-3,-10,1,2,10,-8,-3,7,7,10,10,-1,-10,6,-6,-5,-3,9,5,-9,-3,-3,10,-4,3,-10,1,-7,6,9,2,-8,2,-6,7,3,2],[8,4,-9,-8,-10,-2,10,-4,9,4,10,-3,3,4,3,2,8,-3,-4,9,10,3,-1,-8,3,-6,-9,1,-2,-10,1,-4,9,-7,2,9,7,-1,-10,-9,-4,8,-3,-5,-6,-9,6,-7,5,-10,-1,-7,4,9,1,-9,-2,-8,9,-1,8,-3,3,9,-5,-3,9,-7,1,7,1,-7,-6,-9,8,-10,-7,-1,3,8,-1,4,-2,3,6,-5,7,-1,7,-7,-7,-3,2,-5,-6,-2,-4,8,-9,9,-10,-4,4,-1,-3,-8,-7,-7,5,-1,4,9,-4,-6,-6,10,-7,5,10,-4,4,5,3,10,3,8,-7,-9,-3,-7,8,-5,2,10,-7,6,8,-5,-5,-8,6,-10,-4,7,-2,6,-2,6,-4,5,-1,-5,4,2,6,-10,8,9,-6,1,1,-7,10,-6,-2,-6,-3,3]], dtype = "uint16")#candidate|9457|(2, 168)|const|uint16
var_9458 = relay.var("var_9458", dtype = "int8", shape = (729, 1))#candidate|9458|(729, 1)|var|int8
call_9454 = relay.TupleGetItem(func_5572_call(relay.reshape(var_9455.astype('int8'), [8, 6, 11]), relay.reshape(var_9456.astype('float32'), [792,]), relay.reshape(const_9457.astype('uint16'), [336,]), relay.reshape(var_9458.astype('int8'), [729,]), ), 0)
call_9459 = relay.TupleGetItem(func_5577_call(relay.reshape(var_9455.astype('int8'), [8, 6, 11]), relay.reshape(var_9456.astype('float32'), [792,]), relay.reshape(const_9457.astype('uint16'), [336,]), relay.reshape(var_9458.astype('int8'), [729,]), ), 0)
func_4105_call = mod.get_global_var('func_4105')
func_4109_call = mutated_mod.get_global_var('func_4109')
var_9481 = relay.var("var_9481", dtype = "float64", shape = (9, 4))#candidate|9481|(9, 4)|var|float64
call_9480 = func_4105_call(relay.reshape(var_9481.astype('float64'), [12, 3, 1]), relay.reshape(var_9481.astype('float64'), [12, 3, 1]), )
call_9482 = func_4105_call(relay.reshape(var_9481.astype('float64'), [12, 3, 1]), relay.reshape(var_9481.astype('float64'), [12, 3, 1]), )
uop_9496 = relay.tan(const_9457.astype('float64')) # shape=(2, 168)
func_8428_call = mod.get_global_var('func_8428')
func_8429_call = mutated_mod.get_global_var('func_8429')
call_9505 = relay.TupleGetItem(func_8428_call(), 0)
call_9506 = relay.TupleGetItem(func_8429_call(), 0)
func_5178_call = mod.get_global_var('func_5178')
func_5179_call = mutated_mod.get_global_var('func_5179')
call_9515 = relay.TupleGetItem(func_5178_call(), 0)
call_9516 = relay.TupleGetItem(func_5179_call(), 0)
uop_9519 = relay.sinh(const_9457.astype('float64')) # shape=(2, 168)
output = relay.Tuple([call_9452,call_9454,var_9455,var_9456,var_9458,call_9480,var_9481,uop_9496,call_9505,call_9515,uop_9519,])
output2 = relay.Tuple([call_9453,call_9459,var_9455,var_9456,var_9458,call_9482,var_9481,uop_9496,call_9506,call_9516,uop_9519,])
func_9525 = relay.Function([var_9455,var_9456,var_9458,var_9481,], output)
mod['func_9525'] = func_9525
mod = relay.transform.InferType()(mod)
mutated_mod['func_9525'] = func_9525
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9525_call = mutated_mod.get_global_var('func_9525')
var_9527 = relay.var("var_9527", dtype = "int8", shape = (528,))#candidate|9527|(528,)|var|int8
var_9528 = relay.var("var_9528", dtype = "float32", shape = (792,))#candidate|9528|(792,)|var|float32
var_9529 = relay.var("var_9529", dtype = "int8", shape = (729, 1))#candidate|9529|(729, 1)|var|int8
var_9530 = relay.var("var_9530", dtype = "float64", shape = (9, 4))#candidate|9530|(9, 4)|var|float64
call_9526 = func_9525_call(var_9527,var_9528,var_9529,var_9530,)
output = call_9526
func_9531 = relay.Function([var_9527,var_9528,var_9529,var_9530,], output)
mutated_mod['func_9531'] = func_9531
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5416_call = mod.get_global_var('func_5416')
func_5418_call = mutated_mod.get_global_var('func_5418')
call_9552 = relay.TupleGetItem(func_5416_call(), 0)
call_9553 = relay.TupleGetItem(func_5418_call(), 0)
const_9559 = relay.const([[[-8.503900,-6.638738,3.591208,-6.862983,5.593581,-5.725706,4.367250,-7.022465,-5.784401,-8.060136],[4.912727,9.331320,-1.015552,4.165104,8.295099,-8.849927,-0.664265,1.498483,-8.179471,-1.523977]],[[-5.707756,-1.757875,-4.326981,4.130453,8.137691,9.434096,-8.871742,-7.516200,4.181465,0.665563],[-5.247705,2.299715,8.548037,-2.000339,-2.796398,1.351185,-6.237747,4.334812,-2.763280,5.597702]],[[-1.848417,-5.389889,-0.172439,1.973046,-9.720081,-0.916761,1.636680,-3.682835,-3.700857,-0.700954],[6.387756,2.168851,5.421120,7.426090,-5.968977,5.194042,4.562095,6.900024,-1.144657,4.726061]],[[9.658717,-5.749869,3.876231,4.569760,8.915955,2.465690,-4.895949,-5.857622,5.284549,7.106626],[9.453682,-6.355985,3.881427,-4.501019,-3.952086,-7.047346,9.788245,-4.751117,8.172628,-4.782748]],[[8.743813,-6.900213,1.728401,-2.156377,-8.146345,-0.113832,-5.444877,-4.263645,-2.572359,-7.826170],[-2.161124,-0.012571,7.861966,7.816611,5.103314,8.623284,5.231098,-5.616059,7.130859,1.984485]],[[-8.344844,7.766261,-8.467313,-0.239782,-9.732990,7.721761,-4.664173,-3.960753,0.722203,-7.023449],[3.551882,7.910167,3.175281,5.263505,-3.227900,-7.580927,-8.046917,-7.221632,8.961715,2.641498]],[[1.993171,5.545939,9.054421,9.412444,1.878284,7.946880,0.303815,1.750479,-0.454949,5.079615],[2.391050,6.945091,1.606263,-3.841564,7.774952,-5.406415,-8.197641,-0.468135,4.724532,-2.090473]],[[1.585695,-2.761232,3.145689,-7.327203,-4.895329,-5.308012,4.932587,-6.773055,7.636319,-3.600961],[-3.792325,9.733254,-2.421930,-8.600411,5.815152,-7.586273,-2.161898,-6.421146,-2.760167,7.721254]],[[-6.317391,-1.199431,-0.507500,5.595151,0.117606,-7.392407,-9.164855,-2.454887,2.254748,7.515444],[-8.584965,4.870926,-1.831621,-0.893712,-3.617405,5.218279,9.962351,8.361063,-2.495521,-3.148477]],[[-3.958635,-2.095536,-1.915669,5.189926,2.791430,2.008500,-8.850003,5.602232,-1.708633,5.619620],[8.016835,1.904506,-2.844708,-1.882352,2.679098,-4.134966,5.883484,0.239543,3.451805,8.505547]],[[3.254735,-8.926513,-1.184642,-8.844657,8.943515,2.257780,-1.242255,-2.912010,3.254367,-1.222037],[0.674490,-3.918795,3.303125,-7.661115,-4.684980,8.039901,-5.167942,1.024038,4.516220,2.515765]],[[6.988117,-5.198910,-4.283673,-3.840646,6.161006,-9.183099,6.414830,-6.625215,-4.872603,-2.956434],[-6.126621,-7.300599,7.311914,6.377294,1.362684,-3.476352,-0.422905,2.064219,5.082355,-9.057805]],[[-8.687626,5.559773,-9.383235,-5.558994,-4.659125,1.348184,-8.953272,-4.641099,-1.320628,4.223520],[-5.466455,-1.445490,-5.418975,0.777579,-2.978545,-6.398781,0.430442,-7.143303,0.063218,-8.166358]],[[0.534870,4.319685,-1.772234,-9.723222,3.758848,-4.531731,8.418496,-5.311358,-8.976382,0.014889],[-5.905127,6.561810,0.218705,4.484887,-3.869934,-7.951198,8.666721,-0.884473,3.091825,-3.561906]]], dtype = "float32")#candidate|9559|(14, 2, 10)|const|float32
bop_9560 = relay.bitwise_xor(call_9552.astype('int32'), relay.reshape(const_9559.astype('int32'), relay.shape_of(call_9552))) # shape=(14, 2, 10)
bop_9563 = relay.bitwise_xor(call_9553.astype('int32'), relay.reshape(const_9559.astype('int32'), relay.shape_of(call_9553))) # shape=(14, 2, 10)
func_8684_call = mod.get_global_var('func_8684')
func_8687_call = mutated_mod.get_global_var('func_8687')
const_9565 = relay.const([True,False,True,False,False,False,False,True,True,False,False,True,False,False,True,True,False,True,True,False,False,True,True,True,True,False,True,False,False,True,False,True,False,True,True,False,True,True,True,False,False,True,True,False,True,False,True,False,True,False,False,False,True,True,True,False,True,False,False,True,True,False,False,True,False,False,True,True,True,False,False,True,False,True,True,True,False,False,True,True,False,True,True,False,False,False,False,True,False,True,False,False,True,True,True,True,True,True,True,False,True,True,False,True,False,True,True,False,False,False,False,False,False,True,False,False,False,True,False,False,False,False,False,True,True,False,True,False,True,False,True,False,True,True,False,True,False,True,False,False,True,False,True,True,True,True,False,True,False,False,False,False,False,False,False,False,False,True,False,True,True,True,False,False,False,False,True,True,True,True,True,False,True,False,False,True,False,False,False,False,False,False,True,False,True,True,False,True,False,True,True,True,False,True,True,True,True,False,False,False,False,True,False,True,False,False,False,False,False,True,False,False,True,True,True,False,False,True,False,False,False,True,True,True,False,False,False,True,True,False,True,True,True,True,False,True,False,True,True,False,False,False,True,True,True,False,False,False,True,False,False,False,True,False,True,True,False,True,False,True,True,True,False,False,False,True,False,False,False,True,True,False,True,False,True,True,False,True,True,True,True,True,True,False,True,False,True,False,True,True,True,False,False,False,True,True,True,True,True,True,False,False,True,False,False,False,True,False,False,False,False,True,True,True,False,True,False,True,True,True,True,False,False,True,False,True,False,True,False,False,False,False,False,True,False,True,False,True,True,False,True,True,True,True,True,True,False,True,True,True,False,False,True,False,True,False,True,False,True,True,False,False,False,False,True,False,True,True,False,True,True,True,False,True,False,True,False,False,False,True,True,True,True,True,True,True,True,True,True,False,False,True,False,False,True,False,True,True,True,True,False,True,False,False,False,True,False,False,True,False,False,True,True,True,False,True,False,True,False,True,True,False,False,False,True,False,False,False,True,True,True,True,False,True,True,False,False,True,True,False,False,True,True,True,False,True,False,True,False,False,False,True,True,True,True,False,True,True,False,False,True,True,True,True,False,False,False,True,False,True,True,False,False,False,True,False,True,False,True,True,False,False,False,True,True,False,True,False,True,True,False,False,False,True,False,True,True,False,False,True,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,False,True,False,False,True,False,False,True,False,True,True,True,False,True,True,False,False,False,True,True,True,False,True,True,True,False,False,True,True,False,False,True,True,False,True,False,False,True,False,True,False,True,False,False,True,False,False,True,False,False,True,False,False,False,True,False,False,False,True,True,False,True,False,True,True,True,False,False,False,False,True,False,True,True,False,False,True,True,False,False,True,True,False,True,True,True,True,True,True,False,False,True,True,False,True,True,True,True,True,True,True,False,True,False,True,True,True,True,False,True,True,True,False,True,True,True,False,False,False,False,True,False,True,False,False,False,True,True,True,False,False,False,False,True,True,False,False,False,True,True,True,False,False,True,True,True,False,True,False,False,True,True,False,True,True,False,True,True,True,True,True,True,True,False,False,False,False,False,False,True,True,True,True,False,False,False,True,False,True,True,False,True,False,False,False,False,False,True,True,True,False,False,False,False,False,False,False,False,True,True,True,False,False,False,False], dtype = "bool")#candidate|9565|(720,)|const|bool
var_9566 = relay.var("var_9566", dtype = "float64", shape = (1280,))#candidate|9566|(1280,)|var|float64
call_9564 = relay.TupleGetItem(func_8684_call(relay.reshape(const_9565.astype('bool'), [720,]), relay.reshape(var_9566.astype('float64'), [1280,]), ), 2)
call_9567 = relay.TupleGetItem(func_8687_call(relay.reshape(const_9565.astype('bool'), [720,]), relay.reshape(var_9566.astype('float64'), [1280,]), ), 2)
var_9573 = relay.var("var_9573", dtype = "float64", shape = (1280,))#candidate|9573|(1280,)|var|float64
bop_9574 = relay.logical_and(var_9566.astype('bool'), relay.reshape(var_9573.astype('bool'), relay.shape_of(var_9566))) # shape=(1280,)
bop_9585 = relay.maximum(bop_9560.astype('int16'), relay.reshape(const_9559.astype('int16'), relay.shape_of(bop_9560))) # shape=(14, 2, 10)
bop_9588 = relay.maximum(bop_9563.astype('int16'), relay.reshape(const_9559.astype('int16'), relay.shape_of(bop_9563))) # shape=(14, 2, 10)
output = relay.Tuple([call_9564,const_9565,bop_9574,bop_9585,])
output2 = relay.Tuple([call_9567,const_9565,bop_9574,bop_9588,])
func_9599 = relay.Function([var_9566,var_9573,], output)
mod['func_9599'] = func_9599
mod = relay.transform.InferType()(mod)
mutated_mod['func_9599'] = func_9599
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9599_call = mutated_mod.get_global_var('func_9599')
var_9601 = relay.var("var_9601", dtype = "float64", shape = (1280,))#candidate|9601|(1280,)|var|float64
var_9602 = relay.var("var_9602", dtype = "float64", shape = (1280,))#candidate|9602|(1280,)|var|float64
call_9600 = func_9599_call(var_9601,var_9602,)
output = call_9600
func_9603 = relay.Function([var_9601,var_9602,], output)
mutated_mod['func_9603'] = func_9603
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6277_call = mod.get_global_var('func_6277')
func_6279_call = mutated_mod.get_global_var('func_6279')
call_9617 = relay.TupleGetItem(func_6277_call(), 0)
call_9618 = relay.TupleGetItem(func_6279_call(), 0)
output = call_9617
output2 = call_9618
func_9627 = relay.Function([], output)
mod['func_9627'] = func_9627
mod = relay.transform.InferType()(mod)
mutated_mod['func_9627'] = func_9627
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9627_call = mutated_mod.get_global_var('func_9627')
call_9628 = func_9627_call()
output = call_9628
func_9629 = relay.Function([], output)
mutated_mod['func_9629'] = func_9629
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8623_call = mod.get_global_var('func_8623')
func_8625_call = mutated_mod.get_global_var('func_8625')
call_9655 = relay.TupleGetItem(func_8623_call(), 0)
call_9656 = relay.TupleGetItem(func_8625_call(), 0)
output = relay.Tuple([call_9655,])
output2 = relay.Tuple([call_9656,])
func_9659 = relay.Function([], output)
mod['func_9659'] = func_9659
mod = relay.transform.InferType()(mod)
mutated_mod['func_9659'] = func_9659
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9659_call = mutated_mod.get_global_var('func_9659')
call_9660 = func_9659_call()
output = call_9660
func_9661 = relay.Function([], output)
mutated_mod['func_9661'] = func_9661
mutated_mod = relay.transform.InferType()(mutated_mod)
var_9685 = relay.var("var_9685", dtype = "int16", shape = (13, 8, 2))#candidate|9685|(13, 8, 2)|var|int16
const_9686 = relay.const([[[-8,-8],[-7,7],[-2,6],[-5,6],[10,-8],[9,-1],[-4,5],[4,9]],[[-1,10],[-1,9],[-6,-7],[10,10],[-3,4],[-4,5],[10,1],[-7,7]],[[9,-5],[6,2],[-9,-4],[-2,8],[3,-9],[1,-5],[5,-4],[-8,2]],[[10,-8],[3,8],[-8,-2],[7,8],[-4,-8],[-3,5],[-1,-2],[-9,8]],[[4,-6],[7,1],[-6,-6],[3,-4],[-10,-1],[7,3],[-4,-9],[-1,-2]],[[5,4],[-2,7],[-9,3],[-8,-8],[-1,-3],[6,8],[-1,3],[-6,-7]],[[-3,-4],[7,2],[-6,-8],[6,-5],[7,-7],[4,-1],[4,6],[-10,-2]],[[-5,7],[10,8],[8,2],[-1,1],[5,-9],[5,5],[-7,-9],[6,-10]],[[-10,3],[-8,-5],[6,-10],[-4,-5],[1,1],[5,5],[-9,-8],[8,5]],[[7,-7],[10,1],[1,3],[4,-4],[3,3],[1,1],[-4,4],[9,1]],[[1,8],[2,-9],[3,-6],[-1,2],[-8,-1],[4,-1],[-3,10],[6,5]],[[-9,-9],[4,-2],[10,-4],[2,5],[2,1],[10,-2],[-6,-2],[-4,-10]],[[-1,6],[6,2],[7,7],[3,-1],[7,-9],[-3,7],[-7,-2],[7,-6]]], dtype = "int16")#candidate|9686|(13, 8, 2)|const|int16
bop_9687 = relay.equal(var_9685.astype('bool'), relay.reshape(const_9686.astype('bool'), relay.shape_of(var_9685))) # shape=(13, 8, 2)
func_6517_call = mod.get_global_var('func_6517')
func_6519_call = mutated_mod.get_global_var('func_6519')
call_9691 = relay.TupleGetItem(func_6517_call(), 0)
call_9692 = relay.TupleGetItem(func_6519_call(), 0)
func_6751_call = mod.get_global_var('func_6751')
func_6753_call = mutated_mod.get_global_var('func_6753')
call_9718 = func_6751_call()
call_9719 = func_6751_call()
output = relay.Tuple([bop_9687,call_9691,call_9718,])
output2 = relay.Tuple([bop_9687,call_9692,call_9719,])
func_9722 = relay.Function([var_9685,], output)
mod['func_9722'] = func_9722
mod = relay.transform.InferType()(mod)
mutated_mod['func_9722'] = func_9722
mutated_mod = relay.transform.InferType()(mutated_mod)
var_9723 = relay.var("var_9723", dtype = "int16", shape = (13, 8, 2))#candidate|9723|(13, 8, 2)|var|int16
func_9722_call = mutated_mod.get_global_var('func_9722')
call_9724 = func_9722_call(var_9723)
output = call_9724
func_9725 = relay.Function([var_9723], output)
mutated_mod['func_9725'] = func_9725
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6366_call = mod.get_global_var('func_6366')
func_6368_call = mutated_mod.get_global_var('func_6368')
call_9727 = relay.TupleGetItem(func_6366_call(), 0)
call_9728 = relay.TupleGetItem(func_6368_call(), 0)
output = call_9727
output2 = call_9728
func_9750 = relay.Function([], output)
mod['func_9750'] = func_9750
mod = relay.transform.InferType()(mod)
output = func_9750()
func_9751 = relay.Function([], output)
mutated_mod['func_9751'] = func_9751
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7737_call = mod.get_global_var('func_7737')
func_7738_call = mutated_mod.get_global_var('func_7738')
call_9771 = func_7737_call()
call_9772 = func_7737_call()
uop_9776 = relay.erf(call_9771.astype('float64')) # shape=(660,)
uop_9778 = relay.erf(call_9772.astype('float64')) # shape=(660,)
var_9790 = relay.var("var_9790", dtype = "float64", shape = (660,))#candidate|9790|(660,)|var|float64
bop_9791 = relay.greater(uop_9776.astype('bool'), relay.reshape(var_9790.astype('bool'), relay.shape_of(uop_9776))) # shape=(660,)
bop_9794 = relay.greater(uop_9778.astype('bool'), relay.reshape(var_9790.astype('bool'), relay.shape_of(uop_9778))) # shape=(660,)
func_6751_call = mod.get_global_var('func_6751')
func_6753_call = mutated_mod.get_global_var('func_6753')
call_9796 = func_6751_call()
call_9797 = func_6751_call()
func_9722_call = mod.get_global_var('func_9722')
func_9725_call = mutated_mod.get_global_var('func_9725')
const_9811 = relay.const([[-9,9,-6,-2,2,-7,-2,9,6,-3,-8,9,10,-5,-6,-10,-7,-2,10,5,1,2,7,-3,7,-9,4,8,-1,-6,5,-1,-6,7,-6,10,4,-1,5,-7,9,-7,5,-8,10,5,-1,-10,-9,3,-9,3,4,8,9,4,-1,-2,6,-5,3,6,6,8,-2,3,5,7,-3,1,-10,9,-5,8,-2,-2,4,-8,1,-8,-9,4,-8,-3,6,10,-10,7,-3,-7,-1,9,-4,-6,4,-9,4,7,6,10,8,10,-2,-1,4,-7,6,10,2,6,7,-1,9,-2,9,-7,-3,-3,3,-7,2,4,9,-8,10,-9,9,10,7,5,4,-3,10,-2,-9,-2,3,9,-1,-7,4,2,7,3,1,-1,1,-2,9,-4,2,-9,3,9,8,-9,-3,6,-10,-3,9,10,8,-1,-7,-5,-2,-4,-6,-2,-10,-7,-9,-2,-5,4,10,6,-5,7,1,-10,10,-9,-8,1,2,-10,4,-2,8,-3,-9,8,-5,7,-9,-10,-1,-5,1,-5,-7,-4,7,-10,-5,5]], dtype = "int16")#candidate|9811|(1, 208)|const|int16
call_9810 = relay.TupleGetItem(func_9722_call(relay.reshape(const_9811.astype('int16'), [13, 8, 2])), 1)
call_9812 = relay.TupleGetItem(func_9725_call(relay.reshape(const_9811.astype('int16'), [13, 8, 2])), 1)
func_7136_call = mod.get_global_var('func_7136')
func_7137_call = mutated_mod.get_global_var('func_7137')
call_9826 = func_7136_call()
call_9827 = func_7136_call()
output = relay.Tuple([bop_9791,call_9796,call_9810,const_9811,call_9826,])
output2 = relay.Tuple([bop_9794,call_9797,call_9812,const_9811,call_9827,])
func_9854 = relay.Function([var_9790,], output)
mod['func_9854'] = func_9854
mod = relay.transform.InferType()(mod)
mutated_mod['func_9854'] = func_9854
mutated_mod = relay.transform.InferType()(mutated_mod)
var_9855 = relay.var("var_9855", dtype = "float64", shape = (660,))#candidate|9855|(660,)|var|float64
func_9854_call = mutated_mod.get_global_var('func_9854')
call_9856 = func_9854_call(var_9855)
output = call_9856
func_9857 = relay.Function([var_9855], output)
mutated_mod['func_9857'] = func_9857
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8578_call = mod.get_global_var('func_8578')
func_8579_call = mutated_mod.get_global_var('func_8579')
call_9870 = relay.TupleGetItem(func_8578_call(), 0)
call_9871 = relay.TupleGetItem(func_8579_call(), 0)
uop_9916 = relay.tan(call_9870.astype('float64')) # shape=(11, 16, 9)
uop_9918 = relay.tan(call_9871.astype('float64')) # shape=(11, 16, 9)
output = relay.Tuple([uop_9916,])
output2 = relay.Tuple([uop_9918,])
func_9925 = relay.Function([], output)
mod['func_9925'] = func_9925
mod = relay.transform.InferType()(mod)
mutated_mod['func_9925'] = func_9925
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9925_call = mutated_mod.get_global_var('func_9925')
call_9926 = func_9925_call()
output = call_9926
func_9927 = relay.Function([], output)
mutated_mod['func_9927'] = func_9927
mutated_mod = relay.transform.InferType()(mutated_mod)
const_9932 = relay.const([[[-8.237081,-9.788008,2.749501,-4.851757,0.597409,-4.467516,7.457031,-8.292466],[6.876832,-3.679913,-7.111286,-1.598744,-5.839624,0.872261,5.523607,8.824721],[3.555139,6.691229,0.749706,3.746469,8.702732,-7.792739,8.367652,-7.225206],[-6.035174,-3.062290,4.211720,7.777833,-3.834704,2.612328,7.788253,-6.133939],[-3.902533,8.893971,9.931399,1.612937,-6.596406,0.963747,-9.938734,-8.496297],[3.433710,6.678025,-0.114419,-1.141942,5.644287,-3.638052,-2.335497,7.284916],[-5.715408,-3.105457,8.414789,5.156447,-5.873866,-6.907000,-2.512939,3.745930],[5.103597,-7.884983,6.791945,4.473345,-1.679238,-2.789574,-8.465888,-0.023100],[-2.878862,-6.769696,-0.217257,-5.276208,-5.769489,4.110815,4.359265,-0.617380]],[[8.752876,-8.320926,-2.436972,8.843321,-4.797854,0.434407,7.246423,4.210815],[0.410767,-6.837917,5.048946,-5.505535,6.175437,-1.639284,-5.731930,4.595267],[-8.279565,8.174833,8.132800,0.907078,-4.001162,-6.795669,-1.755412,2.887670],[9.383363,1.070237,-4.615244,0.545880,7.757703,0.143078,-3.742876,0.557265],[-2.388977,3.582776,6.439133,3.623938,-0.704833,-3.674619,-3.141430,-8.396279],[4.741625,-6.370489,3.093984,2.087523,-0.601407,-8.931763,8.512660,4.529625],[-8.880495,-6.032919,7.094147,4.165850,5.167584,8.731254,-8.088311,-0.159685],[0.793870,5.042175,-7.400823,-1.957541,5.201513,0.873188,-7.271861,3.398504],[-6.344933,3.789023,9.203262,5.908234,-3.584553,-2.210321,-5.483426,-2.713239]],[[-0.559865,-7.109185,-8.406321,9.119251,6.240618,-6.847467,1.833478,4.695342],[1.212050,9.664302,5.599935,-8.071178,4.736470,-5.194327,4.552151,-8.124300],[1.211979,-1.153792,8.437657,5.159836,-6.289825,-1.235352,2.772034,7.978005],[-6.107026,-1.681176,5.093371,0.730215,4.972764,-2.861616,-8.322081,-9.071276],[1.202802,3.245904,-6.620247,-2.497300,2.076224,-7.428307,-6.033445,0.806226],[6.331407,-0.263184,-5.263794,4.045289,-0.821879,-8.330942,-4.396668,-3.175372],[8.848636,-5.196344,-8.433925,1.342829,-7.472374,1.366407,0.005912,2.505413],[-9.227988,-9.326777,-4.896288,-5.049394,5.639997,7.215002,-6.185268,-8.454915],[-1.000977,7.765067,6.710257,-9.633729,9.635738,-1.764435,-7.599457,5.331358]],[[9.578836,-7.750599,6.066739,-2.191869,5.671936,2.590343,8.022049,-6.249120],[-0.279404,6.362130,-1.943273,3.038131,4.021528,-0.941514,-5.216573,-6.574941],[9.366857,8.285056,2.527594,-0.101240,8.825486,-1.888540,3.065966,-8.024993],[1.522394,3.367194,0.094835,8.984771,-8.960845,8.627253,4.856967,-0.051616],[-8.205119,2.924354,-8.915629,-2.325642,8.887173,-8.560956,2.359346,1.109654],[-9.112859,-0.942164,8.996385,-3.702068,-5.260101,-1.732446,4.786260,-8.634223],[-5.300499,-8.886690,5.691495,5.261043,0.073025,7.913297,-7.006959,6.171084],[0.512391,5.806044,7.249100,-8.489213,9.688099,2.034981,-3.575797,-2.208197],[-0.377554,4.373185,4.508574,-3.062382,-7.055058,-1.956727,-3.329039,-9.624573]],[[7.761141,5.150734,2.350158,-6.266761,-5.381646,-6.360995,-0.929988,-1.540284],[7.198370,-7.195489,-6.087648,3.951229,6.983671,4.612846,-5.524546,6.252306],[7.630740,-2.425399,2.960791,1.572935,4.513417,-9.785008,-1.527627,-9.710381],[-3.662063,8.049516,2.090536,7.365131,-8.922391,0.080386,-7.418566,-5.022833],[6.432940,0.225718,-7.559927,-3.390946,-7.365687,5.818410,1.204357,-6.160016],[-6.455295,-2.155695,-1.312321,-9.369456,7.394110,5.974335,9.998948,3.149365],[0.218771,9.365457,-3.196540,6.057592,-3.810764,7.302170,-2.031183,8.247590],[2.878243,-3.708592,6.611661,-6.673007,-9.946539,1.762594,-1.258491,-1.229030],[7.731648,4.551644,-1.988998,-4.678478,7.920945,-9.641551,3.172913,-7.822859]],[[-0.315901,-7.850992,-5.499697,-0.882539,7.287255,-8.028336,-8.701841,6.410035],[-8.592355,8.141834,6.054041,2.435815,-8.905331,-9.691234,6.225914,-0.081109],[7.049249,0.559410,-6.986507,1.232946,3.198298,0.754130,-2.559632,3.078225],[1.332565,3.313790,-8.164839,9.704596,7.480410,-9.351567,-2.394207,-8.510294],[-9.732925,9.973261,2.450554,-7.583065,-2.104462,-9.073127,-8.761543,1.229569],[8.232557,-8.909435,6.958184,-5.751504,6.587052,2.386461,3.075490,1.835798],[-9.574919,-7.477306,8.697547,-9.961903,-6.661120,2.532275,4.565554,-9.601629],[4.842730,-1.126523,-9.762448,9.604643,-1.184424,-0.373244,-7.026236,0.351271],[-7.988090,-0.207385,-1.838555,-3.967157,-7.822044,6.387897,-5.257538,-2.810605]],[[-9.140932,1.462193,6.805834,-1.885956,5.743331,1.986670,-7.183696,-9.614244],[1.749580,6.591742,-0.438559,-2.606818,9.300750,-9.127989,-0.905602,3.917225],[2.914988,9.419025,-2.872653,-8.280031,-4.616403,8.945927,-6.034907,-1.331630],[2.462470,5.581291,1.263613,-2.245042,8.974912,-4.547484,5.326866,-3.691977],[-9.249776,4.270894,3.398270,1.874357,7.428394,5.196896,-0.102071,1.297506],[3.780315,8.307988,9.055955,4.295424,-7.379041,-3.798719,5.268477,9.543412],[-2.159079,-8.030551,-8.502577,-4.969673,8.557055,-1.529166,2.935625,3.973519],[-1.059123,8.124242,-2.167917,8.027262,-2.467068,-6.728988,-4.291754,-7.251566],[-5.594606,6.130238,-8.458389,-5.261636,9.751150,1.506131,6.034958,9.307864]],[[-7.434213,3.384410,3.214700,6.281889,-9.690940,1.333473,0.856606,1.263922],[5.218713,6.388584,-6.396769,8.349496,-7.630957,-2.538494,8.046794,9.109615],[-2.142759,-8.691095,7.667159,-0.633473,9.215822,1.606221,1.904957,-2.986125],[7.626568,-2.447286,7.516456,2.103303,-5.927028,1.740996,7.928204,8.152348],[3.945050,7.808541,-1.222991,-3.627662,3.609744,-7.055624,3.396531,5.631649],[-4.876014,-8.723503,1.729619,0.802024,-0.389873,0.809222,-8.349699,7.774345],[-5.435917,-4.787098,-1.862099,-6.477019,-9.005447,-6.561589,1.438326,-8.730374],[3.789070,-8.659313,-9.145513,9.717346,-6.512981,-5.942914,4.937593,1.206103],[-4.481227,-4.382701,-4.963534,1.555738,-9.523258,-5.426245,8.520727,0.278266]],[[-4.156038,-0.588681,-4.109059,5.586698,-3.868951,6.366153,4.243004,-0.385055],[-3.057023,0.624259,8.191299,-9.834152,-8.965302,-3.690475,3.858877,4.177305],[1.234236,3.139719,8.571295,-5.071624,-1.221039,5.947610,-8.204051,-3.646234],[2.553120,-1.740272,-3.609567,-9.351791,5.588770,-1.004209,-8.492504,-8.560797],[1.868024,7.417222,-0.713761,1.457238,9.970357,9.426531,-5.637877,7.651127],[3.597254,-2.583028,-4.792124,-5.418323,7.867535,-3.242556,-8.666762,5.280169],[-3.309745,-2.136737,6.850756,-4.346359,4.130676,-9.601474,8.181777,7.622980],[0.291947,-1.421345,-7.976689,-1.521490,3.897535,4.949452,2.396949,-9.092276],[-6.474725,3.687472,-1.337621,-1.979993,3.180444,8.173245,-4.333706,7.971803]],[[-5.671754,9.491883,-7.824250,0.903731,6.896144,-5.484784,9.801489,-5.871074],[2.078002,0.164172,2.685235,8.310648,5.833448,7.654926,2.471574,3.599953],[-1.726689,-8.882991,-8.504645,3.832040,4.496000,5.642021,-7.025702,-4.619006],[0.575077,8.997285,9.534052,8.454762,6.194969,7.358911,3.385165,-9.789616],[1.764553,-5.620132,-4.690595,3.591799,7.827496,3.672435,3.787297,-9.753030],[2.146724,8.966725,-4.701779,5.805752,-2.569720,7.485759,-0.526211,8.233429],[-8.101662,0.562566,0.476136,-5.892389,4.416599,-1.090073,1.503715,2.192527],[-0.056719,-4.770560,4.187297,-6.833517,-1.383283,8.081154,-5.881097,-8.414377],[4.660999,9.033366,3.667607,2.570913,5.752772,9.971165,-6.658327,-7.933743]],[[0.039543,-5.624811,-7.290552,-7.346969,-0.904567,2.982254,8.192398,1.701188],[-8.554903,7.910233,7.814631,-7.832391,4.630515,3.332427,5.616887,-6.164487],[-3.137026,8.736069,-3.434864,8.201443,-7.724182,3.278086,8.312929,-8.700983],[2.299528,7.525590,-2.700868,-3.704155,0.017540,-3.326531,1.857735,7.429594],[5.890405,-6.828057,7.176924,-2.954657,0.112090,6.771343,-3.522608,-3.637415],[0.032986,-1.821847,8.016296,1.409913,4.613758,0.465859,6.232328,-3.765023],[-8.755317,-7.625367,3.655889,5.140391,-3.056255,2.113269,-4.051722,-2.356548],[6.352504,1.885923,3.942824,6.509552,-0.590534,1.398089,9.144445,-5.018484],[9.572090,1.263457,4.996910,0.913893,9.767489,2.767679,-2.330931,-2.150535]],[[-1.843056,-4.444800,-5.709141,-8.674820,0.177396,-8.528814,-4.154539,-0.161736],[3.497226,-1.802534,-8.345191,-0.722843,-3.804634,4.827638,-7.955540,-6.786746],[6.323791,-4.254783,-1.238761,4.018826,4.970280,-3.851800,-3.113657,1.018132],[4.617294,6.593652,1.377354,2.495653,3.454110,-3.222689,-4.684415,-5.187912],[8.730277,-0.272540,-6.561069,2.935992,9.767896,6.653382,-6.900338,-5.690879],[-4.046706,0.218510,6.138044,8.363767,3.405338,-5.188187,-5.714222,-3.247429],[-0.804621,5.657065,-2.946372,-0.893087,-1.926530,0.149114,-1.645462,-3.664173],[1.925425,-9.549405,1.336584,4.091592,0.691295,-5.116489,1.861377,2.613271],[5.137942,-4.035562,-2.345510,-1.073274,-0.598594,6.424155,7.804029,2.403523]],[[-7.322428,2.093550,4.701939,-5.860889,5.705872,-9.862486,6.929983,-7.869276],[-1.193177,-3.876194,3.504861,0.812003,4.342004,-9.560714,3.133262,7.337362],[2.908602,6.381431,-6.804325,4.712823,-7.327406,-1.317626,1.006893,9.757248],[1.588173,-0.191746,-8.240492,-9.578765,5.652229,-0.083861,-7.991897,-8.867948],[-7.035575,-1.560699,1.038212,-3.985004,2.281222,-9.651100,6.485509,-9.293272],[7.049819,1.272805,-3.893452,3.738499,-3.931679,-5.557905,-3.326852,-8.494034],[-4.430978,9.266671,-1.222507,-4.366430,-5.080231,2.382673,-9.805148,-0.065380],[-7.438162,8.653892,5.766553,-4.392122,-5.158484,-8.844340,9.793995,9.156866],[0.760607,-3.203083,3.702398,9.928966,4.289027,3.164961,-5.990880,2.824382]],[[-3.874998,9.851164,-0.048178,8.702309,4.587178,6.635297,9.422129,-5.562558],[1.021505,-1.216100,8.454551,-2.715736,-1.808959,-4.505945,6.064694,-9.510589],[-2.363521,3.290150,8.363361,4.462595,-0.503655,-9.593345,-3.221899,-8.693463],[1.498161,-7.163910,-5.486782,4.606756,-8.460931,3.517135,2.028177,2.526380],[8.668781,-0.345199,-4.460704,8.649009,-6.279467,-2.468928,9.920025,-2.304055],[7.828706,3.570312,7.759017,4.299609,4.977046,8.939192,-9.087074,-0.548768],[1.077533,-1.962386,4.813085,-5.810684,8.494706,-0.732213,8.332717,7.833602],[7.744470,-6.520380,8.212459,-6.632803,-1.868197,-7.004762,1.299607,7.737292],[2.418691,-1.372810,5.399602,1.500768,8.205208,3.099437,-8.499603,2.414655]],[[-5.237618,5.845276,0.058905,-1.618740,8.733024,-4.943146,9.894226,-1.043442],[-3.974263,8.767955,1.381059,-1.853637,-4.768234,-6.034493,4.061241,-6.624449],[9.678397,1.649537,-8.218156,3.832003,-0.805096,-7.984712,1.612026,5.373239],[9.784474,6.350506,6.234560,5.067169,-6.418400,-4.503398,6.799868,0.388647],[-0.967174,-5.878250,-4.942062,3.152593,0.239209,-7.257232,-5.959088,-5.052282],[2.211499,-4.204924,-1.025779,-0.325686,5.773474,-9.937952,-8.092208,-9.941076],[-2.560085,9.324018,-6.458225,-1.041127,-5.813539,-0.296986,-8.802661,-2.028151],[-4.345184,-2.021266,-6.554833,5.668765,4.545883,-1.037300,1.331399,8.784245],[2.591564,-9.896956,8.695122,-9.523536,-3.099811,-1.181191,8.402484,-9.084822]]], dtype = "float32")#candidate|9932|(15, 9, 8)|const|float32
uop_9933 = relay.atan(const_9932.astype('float32')) # shape=(15, 9, 8)
output = relay.Tuple([uop_9933,])
output2 = relay.Tuple([uop_9933,])
func_9943 = relay.Function([], output)
mod['func_9943'] = func_9943
mod = relay.transform.InferType()(mod)
mutated_mod['func_9943'] = func_9943
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9943_call = mutated_mod.get_global_var('func_9943')
call_9944 = func_9943_call()
output = call_9944
func_9945 = relay.Function([], output)
mutated_mod['func_9945'] = func_9945
mutated_mod = relay.transform.InferType()(mutated_mod)
const_9961 = relay.const([[[-6.035808,4.262098,8.281242,-0.937320,-8.086905,4.990362,1.245053,-9.421755,7.774933,-8.109758,1.594031,3.126467,8.917264,3.838303],[0.216505,9.513485,7.352047,-4.686430,-4.126403,-5.009945,7.748437,-0.412327,-4.925469,-7.818171,5.264577,-7.261389,-7.437368,1.052090],[-7.159126,4.993897,8.787058,2.099074,6.070601,1.026268,-6.638607,-4.863599,9.874002,-5.656674,1.765381,-9.363827,-7.676409,-2.655567]],[[-4.113776,-3.702871,-4.890198,-2.063901,-1.204031,-9.632946,-6.927076,3.590114,-7.766898,-5.337180,6.227961,6.954756,5.533969,1.662002],[-3.144257,-8.476721,5.098377,0.756635,-8.713800,-4.290998,-6.908772,-4.590499,-3.314384,1.461496,-6.150886,5.911813,3.553321,-4.374672],[-2.448960,-1.498904,4.307496,9.947001,7.487689,8.340457,7.657591,4.319261,-0.005512,-9.379692,6.386713,5.598109,9.719612,1.807834]],[[7.919555,7.168513,-8.740207,-7.192531,1.552617,-3.662177,7.956815,-0.797965,5.896800,-3.714136,-3.346331,9.746217,-3.451893,8.175674],[7.059029,-6.031972,-9.601221,-8.519366,9.027963,2.424439,-9.152562,-4.612185,2.653442,-0.047080,-7.115139,-6.341916,3.532131,-7.237115],[6.083546,6.773691,0.397696,9.766099,-1.808402,-0.948961,4.842107,-5.698802,0.934949,7.546515,7.482420,-3.939002,-7.004402,-9.717084]],[[-4.998319,9.738997,-5.515098,-9.139445,9.701619,-5.741613,9.755047,6.304677,-3.580033,-9.758793,-1.088096,-3.720038,-0.976526,3.874188],[3.127478,-0.129053,4.463035,8.831597,4.367773,0.911870,-8.778324,-8.290597,-4.369599,-7.535926,9.747994,-1.686038,-7.156773,0.105807],[8.030897,7.115894,9.429178,-4.419483,5.882611,-0.914537,6.323198,5.415434,6.668014,1.044532,-1.264785,-4.587397,4.033709,4.915828]],[[5.147741,-2.178616,-0.879084,-0.169270,9.722372,-3.920678,-0.872609,3.374519,-3.154932,2.658174,-5.740626,6.938635,-4.836822,-9.842339],[-2.224957,9.322120,5.684515,7.773646,0.756886,0.198874,-9.867245,-5.152788,-1.040854,2.189361,-7.990006,3.767812,6.193125,4.386119],[-4.613382,2.729350,-4.157342,2.419484,-7.767561,7.327636,-3.974051,0.942288,-0.307605,1.743317,2.699504,9.377723,-9.594574,3.526514]],[[5.386510,2.892652,-6.739794,-3.056733,-7.129639,-9.582387,-2.525783,-4.245132,-7.243296,-8.331343,1.858896,-1.281152,-6.048119,7.484291],[-1.996553,-1.486126,1.457125,0.130385,9.409524,-3.051568,5.246120,-0.023691,-1.284174,-2.453177,-5.741457,4.502441,7.701999,3.555882],[1.419660,-1.636838,6.986109,4.294707,-2.210429,-0.403399,9.781646,3.809919,-0.450396,-1.523973,4.845613,3.604956,8.908476,7.439119]],[[-3.072565,-5.072304,7.388024,8.163120,-9.441951,-0.102614,-0.203727,-9.211102,2.187325,-0.275707,7.483703,4.045002,-4.330146,2.522164],[5.368880,-2.656249,4.531025,-5.850177,6.164441,0.359643,-5.903462,-0.595656,-8.696924,-5.803481,2.264442,2.288411,-4.645742,9.637126],[-9.322731,-5.900640,4.454329,4.360327,-9.094741,-0.049944,-2.364103,9.760761,6.586895,2.570662,-6.240741,5.222351,-0.451387,-5.098609]],[[4.187619,4.087813,-3.555011,-6.204228,-7.924822,-1.077691,0.431602,-8.816475,-0.879817,-4.507088,-1.472277,-4.304689,-1.327920,-4.248371],[-3.139292,8.666250,-3.557937,2.524507,-6.221791,-8.349134,-7.063086,1.229171,7.927935,4.913431,7.022356,2.424571,2.556138,-0.396646],[-1.264451,-1.195291,-2.451764,-5.144228,8.234819,4.734237,3.436646,-8.186252,4.674006,8.143177,-4.797123,-1.297076,0.496475,7.419894]],[[7.885579,-3.782398,-5.643187,6.792239,-2.444775,2.607809,3.729910,0.826968,-5.295401,-3.858256,-2.437034,1.220468,-0.568766,-7.389419],[2.350050,-9.239039,-7.620988,-9.910254,-9.695085,5.627492,-4.955499,-8.866997,5.226004,-4.410594,5.324270,3.846985,5.089519,-3.907112],[7.674506,5.751870,-7.872671,5.296611,-5.372326,6.704679,9.140112,1.452032,7.616362,-8.089392,7.298359,9.534614,7.536694,-8.109085]],[[8.571934,4.197079,-5.350417,9.929596,-6.824656,5.575917,7.117065,8.952745,6.797731,-4.593691,-7.549261,0.289154,2.772649,3.697598],[-1.023396,6.879085,-3.848235,-6.916580,5.854238,-3.258580,-6.875310,-4.296730,-6.518231,0.722119,-8.485899,1.035757,-1.377029,3.974553],[5.917290,-5.772817,-9.628862,6.475453,-6.784257,9.475450,2.126944,-8.289178,4.925363,5.504854,-6.183489,2.695568,-4.446429,-4.408839]],[[-0.407396,1.538812,-0.637041,0.308959,6.429550,1.003233,-4.626227,-3.334516,-3.442917,7.813908,2.471794,-1.446674,4.830204,9.680161],[-7.699750,-7.752904,1.256015,-7.896425,-8.321971,-2.032991,-2.961605,8.980573,0.495017,-7.827989,-3.013344,0.466569,6.373898,-3.653897],[-6.475156,-9.861095,5.040025,9.206771,-2.031894,-7.893536,9.894455,2.071096,-3.971781,-0.932013,-9.026386,-4.639226,-0.096837,-4.328727]],[[6.854022,4.004450,-0.268203,-3.953029,-0.299709,-3.981414,2.511662,1.835698,-8.129075,-2.018855,-1.184990,-4.454077,3.779525,4.172779],[-9.963045,5.931671,-3.865244,0.286314,7.059996,-2.626102,9.566091,-9.942159,7.870374,7.161277,-7.601014,5.809439,-1.318679,8.316236],[-3.853004,-4.302308,6.546116,-3.707425,-3.559298,-8.252999,2.199973,4.121356,-1.374046,9.812204,-7.763029,0.651424,-8.180833,9.179770]],[[-8.749625,-6.091685,1.510191,5.657435,6.942993,8.685758,-7.082731,2.857684,-3.725376,-9.881008,-1.247559,0.727720,2.965796,-7.419774],[7.387595,-7.024511,-5.658589,-9.696686,2.029796,-8.387558,7.581395,-2.798606,-6.195939,4.310197,6.528248,4.345098,1.114373,-0.436069],[1.020058,-7.142742,-3.962411,-5.037296,0.141265,-6.525431,-5.502998,7.013131,-5.060505,0.969363,1.812938,-8.450567,4.079450,4.911524]],[[-2.341367,4.259072,-2.894834,-5.505952,-5.131500,0.556967,7.530235,1.314104,6.000296,6.625962,-2.386554,-5.549195,3.110165,-0.493073],[-5.533569,2.921568,4.258822,-1.128323,-4.823634,-6.932746,3.685975,-8.240495,-6.073078,-2.294260,9.746593,9.917527,0.271092,1.785215],[-6.149356,3.971275,2.424475,0.191845,5.555982,-4.409235,-4.372660,-1.349550,3.454216,-6.761809,-8.738452,4.361722,-3.604053,-8.460219]],[[-0.546008,-6.858915,4.156045,6.048560,-8.084793,-9.751345,-6.382168,6.462628,8.543192,-6.529094,6.727645,-4.164264,1.412641,0.055206],[1.817736,8.223165,4.101794,-1.457581,3.804155,-8.394980,6.082670,4.093191,6.247218,-7.131413,3.242247,6.601991,-4.496834,-8.030729],[9.382072,6.404266,6.207093,-6.061999,6.919924,-6.388928,-2.229218,-3.701333,2.193529,6.565116,9.798029,6.691905,-9.994440,5.380529]],[[6.251091,6.744457,9.684591,4.847495,1.728133,1.258571,-7.227137,-4.818644,-1.503001,-0.793710,-8.244286,3.756150,-2.559851,-4.578979],[5.051550,6.396708,-5.108174,-7.396734,6.726324,-4.503883,2.611115,0.251499,8.109842,2.367515,4.810317,4.858941,7.401418,4.767291],[-7.104916,5.093999,6.707930,-0.146974,3.983097,8.489730,-2.050456,3.501432,9.429136,0.106349,-7.844367,6.525185,2.598110,-7.069154]]], dtype = "float64")#candidate|9961|(16, 3, 14)|const|float64
uop_9962 = relay.log10(const_9961.astype('float64')) # shape=(16, 3, 14)
func_8918_call = mod.get_global_var('func_8918')
func_8920_call = mutated_mod.get_global_var('func_8920')
call_9964 = func_8918_call()
call_9965 = func_8918_call()
func_8031_call = mod.get_global_var('func_8031')
func_8033_call = mutated_mod.get_global_var('func_8033')
var_9986 = relay.var("var_9986", dtype = "float64", shape = (660,))#candidate|9986|(660,)|var|float64
call_9985 = relay.TupleGetItem(func_8031_call(relay.reshape(var_9986.astype('float64'), [660,])), 1)
call_9987 = relay.TupleGetItem(func_8033_call(relay.reshape(var_9986.astype('float64'), [660,])), 1)
output = relay.Tuple([uop_9962,call_9964,call_9985,var_9986,])
output2 = relay.Tuple([uop_9962,call_9965,call_9987,var_9986,])
func_9988 = relay.Function([var_9986,], output)
mod['func_9988'] = func_9988
mod = relay.transform.InferType()(mod)
var_9989 = relay.var("var_9989", dtype = "float64", shape = (660,))#candidate|9989|(660,)|var|float64
output = func_9988(var_9989)
func_9990 = relay.Function([var_9989], output)
mutated_mod['func_9990'] = func_9990
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4857_call = mod.get_global_var('func_4857')
func_4858_call = mutated_mod.get_global_var('func_4858')
call_10024 = relay.TupleGetItem(func_4857_call(), 3)
call_10025 = relay.TupleGetItem(func_4858_call(), 3)
func_5968_call = mod.get_global_var('func_5968')
func_5970_call = mutated_mod.get_global_var('func_5970')
call_10026 = func_5968_call()
call_10027 = func_5968_call()
func_6794_call = mod.get_global_var('func_6794')
func_6795_call = mutated_mod.get_global_var('func_6795')
call_10059 = relay.TupleGetItem(func_6794_call(), 0)
call_10060 = relay.TupleGetItem(func_6795_call(), 0)
output = relay.Tuple([call_10024,call_10026,call_10059,])
output2 = relay.Tuple([call_10025,call_10027,call_10060,])
func_10070 = relay.Function([], output)
mod['func_10070'] = func_10070
mod = relay.transform.InferType()(mod)
mutated_mod['func_10070'] = func_10070
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10070_call = mutated_mod.get_global_var('func_10070')
call_10071 = func_10070_call()
output = call_10071
func_10072 = relay.Function([], output)
mutated_mod['func_10072'] = func_10072
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5178_call = mod.get_global_var('func_5178')
func_5179_call = mutated_mod.get_global_var('func_5179')
call_10104 = relay.TupleGetItem(func_5178_call(), 0)
call_10105 = relay.TupleGetItem(func_5179_call(), 0)
output = call_10104
output2 = call_10105
func_10111 = relay.Function([], output)
mod['func_10111'] = func_10111
mod = relay.transform.InferType()(mod)
output = func_10111()
func_10112 = relay.Function([], output)
mutated_mod['func_10112'] = func_10112
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8166_call = mod.get_global_var('func_8166')
func_8168_call = mutated_mod.get_global_var('func_8168')
call_10127 = relay.TupleGetItem(func_8166_call(), 0)
call_10128 = relay.TupleGetItem(func_8168_call(), 0)
output = relay.Tuple([call_10127,])
output2 = relay.Tuple([call_10128,])
func_10133 = relay.Function([], output)
mod['func_10133'] = func_10133
mod = relay.transform.InferType()(mod)
output = func_10133()
func_10134 = relay.Function([], output)
mutated_mod['func_10134'] = func_10134
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9257_call = mod.get_global_var('func_9257')
func_9258_call = mutated_mod.get_global_var('func_9258')
call_10138 = relay.TupleGetItem(func_9257_call(), 0)
call_10139 = relay.TupleGetItem(func_9258_call(), 0)
output = call_10138
output2 = call_10139
func_10179 = relay.Function([], output)
mod['func_10179'] = func_10179
mod = relay.transform.InferType()(mod)
output = func_10179()
func_10180 = relay.Function([], output)
mutated_mod['func_10180'] = func_10180
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7259_call = mod.get_global_var('func_7259')
func_7260_call = mutated_mod.get_global_var('func_7260')
call_10184 = relay.TupleGetItem(func_7259_call(), 0)
call_10185 = relay.TupleGetItem(func_7260_call(), 0)
output = call_10184
output2 = call_10185
func_10200 = relay.Function([], output)
mod['func_10200'] = func_10200
mod = relay.transform.InferType()(mod)
output = func_10200()
func_10201 = relay.Function([], output)
mutated_mod['func_10201'] = func_10201
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6751_call = mod.get_global_var('func_6751')
func_6753_call = mutated_mod.get_global_var('func_6753')
call_10286 = func_6751_call()
call_10287 = func_6751_call()
func_8789_call = mod.get_global_var('func_8789')
func_8791_call = mutated_mod.get_global_var('func_8791')
call_10288 = relay.TupleGetItem(func_8789_call(), 0)
call_10289 = relay.TupleGetItem(func_8791_call(), 0)
output = relay.Tuple([call_10286,call_10288,])
output2 = relay.Tuple([call_10287,call_10289,])
func_10317 = relay.Function([], output)
mod['func_10317'] = func_10317
mod = relay.transform.InferType()(mod)
mutated_mod['func_10317'] = func_10317
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10317_call = mutated_mod.get_global_var('func_10317')
call_10318 = func_10317_call()
output = call_10318
func_10319 = relay.Function([], output)
mutated_mod['func_10319'] = func_10319
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8918_call = mod.get_global_var('func_8918')
func_8920_call = mutated_mod.get_global_var('func_8920')
call_10320 = func_8918_call()
call_10321 = func_8918_call()
func_8578_call = mod.get_global_var('func_8578')
func_8579_call = mutated_mod.get_global_var('func_8579')
call_10349 = relay.TupleGetItem(func_8578_call(), 1)
call_10350 = relay.TupleGetItem(func_8579_call(), 1)
output = relay.Tuple([call_10320,call_10349,])
output2 = relay.Tuple([call_10321,call_10350,])
func_10353 = relay.Function([], output)
mod['func_10353'] = func_10353
mod = relay.transform.InferType()(mod)
output = func_10353()
func_10354 = relay.Function([], output)
mutated_mod['func_10354'] = func_10354
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7461_call = mod.get_global_var('func_7461')
func_7463_call = mutated_mod.get_global_var('func_7463')
call_10374 = relay.TupleGetItem(func_7461_call(), 0)
call_10375 = relay.TupleGetItem(func_7463_call(), 0)
output = relay.Tuple([call_10374,])
output2 = relay.Tuple([call_10375,])
func_10383 = relay.Function([], output)
mod['func_10383'] = func_10383
mod = relay.transform.InferType()(mod)
output = func_10383()
func_10384 = relay.Function([], output)
mutated_mod['func_10384'] = func_10384
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5935_call = mod.get_global_var('func_5935')
func_5936_call = mutated_mod.get_global_var('func_5936')
call_10406 = func_5935_call()
call_10407 = func_5935_call()
output = call_10406
output2 = call_10407
func_10450 = relay.Function([], output)
mod['func_10450'] = func_10450
mod = relay.transform.InferType()(mod)
output = func_10450()
func_10451 = relay.Function([], output)
mutated_mod['func_10451'] = func_10451
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7461_call = mod.get_global_var('func_7461')
func_7463_call = mutated_mod.get_global_var('func_7463')
call_10467 = relay.TupleGetItem(func_7461_call(), 0)
call_10468 = relay.TupleGetItem(func_7463_call(), 0)
output = relay.Tuple([call_10467,])
output2 = relay.Tuple([call_10468,])
func_10470 = relay.Function([], output)
mod['func_10470'] = func_10470
mod = relay.transform.InferType()(mod)
output = func_10470()
func_10471 = relay.Function([], output)
mutated_mod['func_10471'] = func_10471
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6666_call = mod.get_global_var('func_6666')
func_6668_call = mutated_mod.get_global_var('func_6668')
call_10503 = relay.TupleGetItem(func_6666_call(), 0)
call_10504 = relay.TupleGetItem(func_6668_call(), 0)
func_6366_call = mod.get_global_var('func_6366')
func_6368_call = mutated_mod.get_global_var('func_6368')
call_10529 = relay.TupleGetItem(func_6366_call(), 0)
call_10530 = relay.TupleGetItem(func_6368_call(), 0)
func_7442_call = mod.get_global_var('func_7442')
func_7445_call = mutated_mod.get_global_var('func_7445')
var_10541 = relay.var("var_10541", dtype = "float64", shape = (2, 640))#candidate|10541|(2, 640)|var|float64
call_10540 = relay.TupleGetItem(func_7442_call(relay.reshape(var_10541.astype('float64'), [8, 10, 16])), 0)
call_10542 = relay.TupleGetItem(func_7445_call(relay.reshape(var_10541.astype('float64'), [8, 10, 16])), 0)
output = relay.Tuple([call_10503,call_10529,call_10540,var_10541,])
output2 = relay.Tuple([call_10504,call_10530,call_10542,var_10541,])
func_10546 = relay.Function([var_10541,], output)
mod['func_10546'] = func_10546
mod = relay.transform.InferType()(mod)
var_10547 = relay.var("var_10547", dtype = "float64", shape = (2, 640))#candidate|10547|(2, 640)|var|float64
output = func_10546(var_10547)
func_10548 = relay.Function([var_10547], output)
mutated_mod['func_10548'] = func_10548
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6517_call = mod.get_global_var('func_6517')
func_6519_call = mutated_mod.get_global_var('func_6519')
call_10550 = relay.TupleGetItem(func_6517_call(), 0)
call_10551 = relay.TupleGetItem(func_6519_call(), 0)
output = call_10550
output2 = call_10551
func_10552 = relay.Function([], output)
mod['func_10552'] = func_10552
mod = relay.transform.InferType()(mod)
mutated_mod['func_10552'] = func_10552
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10552_call = mutated_mod.get_global_var('func_10552')
call_10553 = func_10552_call()
output = call_10553
func_10554 = relay.Function([], output)
mutated_mod['func_10554'] = func_10554
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9659_call = mod.get_global_var('func_9659')
func_9661_call = mutated_mod.get_global_var('func_9661')
call_10596 = relay.TupleGetItem(func_9659_call(), 0)
call_10597 = relay.TupleGetItem(func_9661_call(), 0)
func_5908_call = mod.get_global_var('func_5908')
func_5912_call = mutated_mod.get_global_var('func_5912')
const_10610 = relay.const([[-7,-9,-1,-3,5,-2,2,6,-10,-3,1,-4,9,4,5,-10,-4,-1,6,-8,9,-7,-6,4,-7,4,4,-8,3,-3,-6,1,2,5,-9,10,2,10,10,4,10,5,8,-1],[7,-7,4,5,1,7,4,7,-3,-4,10,10,5,-8,-1,4,-10,3,9,4,-9,-8,-8,-8,4,6,9,-6,3,7,-2,4,-2,-7,-8,-10,2,9,-5,-1,-9,2,2,2],[1,-1,-2,4,5,-4,-2,9,2,-4,-9,5,9,-5,7,-8,9,-1,6,5,-7,-6,1,5,-7,2,3,3,-5,4,-8,-10,4,9,7,5,3,6,-7,7,9,10,-7,5]], dtype = "uint16")#candidate|10610|(3, 44)|const|uint16
const_10611 = relay.const([-5,4,7,-9,1,-2,-1,-6,-7,8,-6,5,9,9,7,-3,-9,10,-6,6,-4,-4,1,9,-4,6,-4,-8,-5,6,6,-6,6,-3,-2,2,7,-10,3,-6,6,6,3,4,-10,-1,10,1,1,10,5,9,5,10,-9,-6,-8,-9,-8,-10,1,-5,7,7,-4,-3,-9,2,-2,-4,9,-9,2,-10,9,9,1,3,9,-8,-8,-1,5,-1,-6,-9,-4,-3,-4,-7,5,5,9,4,7,-1,2,2,-7,-4,9,-9,-2,7,7,1,9,8,1,6,-3,-4,10,4,-10,-2,4,10,-6,-1,3,5,-7,2,4,-7,6,3,9,3,6,6,7,-3,5,5,8,-3,-4,3,4,4,3,-6,-3,-10,6,-6,-4,-5,-4,8,-2,-7,6,4,-9,-10,6,-5,-6,-9,-4,5,-5,-8,3,-3,9,-3,9,6,8,2,-3,4,-5,4,-7,9,9,2,-5,9,6,-2,-1,-5,-6,4,8,-1,-2,-4,3,-2,-1,4,-3,2,7,6,-2,-9,-10,10,-3,-2,-7,4,10,-2,-4,8,1,7,-4,8,-4,8,9,3,-10,-7,-3,-9,-8,-3,1,-8,-3,-2,5,-5,-6,4,-1,-5,4,-2,-2,-10,-3,-3,-6,7,-7,6,-9,-3,-4,-10,4,9,-7,4,9,-2,-9,-7,7,2,-2,-5,1,-8,5,-1,-3,-8,1,-2,6,-5,10,-9,5,-2,-2,-10,7,8,-5,-2,1,-1,-9,-3,8,1,-9,6,4,10,4,-6,9,-4,7,3,3,-10,-6,1,-10,-1,-3,-2,-4,2,2,1,-6,-4,-1,10,-2,8,3,1,2,-4,-6,3,-7,-9,2,8,5,8,-2,1,9,1,9,-7,1,-5,6,-8,2,1,7,2,5,9,-10,-1,-10,-3,4,7,1,-2,-1,-1,-8,-6,6,-1,10,10,10,1,6,8,-3,-3,10,-3,-8,3,9,-2,-7,-5,3,-9,9,-5,-1,2,10,10,3,-9,8,-8,1,-1,5,7,1,2,-3,4,-10,10,2,-7,3,-7,-10,-7,4,-1,-2,-6,-4,5,-5,-5,-5,-2,7,3,5,-1,2,-6,-4,7,5,2,-2,-9,6,-2,9,2,-3,-5,3,-3,-4,5,-6,-5,-6,10,-4,4,3,2,10,-9,9,1,-2,8,3,-3,6,2,2,9,-3,7,1,-1,3,3,-5,-5,1,10,-9,-5,-10,-4,-10,-10,5,10,-7,6,10,4,-8,-10,-4,-2,-5,7,-5,8,3,-8,-5,6,-7,4,6,1,-6,-10,3,-5,-1,2,1,-1,9,-5,-8,-8,5,-2,-9,4,4,-8,-8,-5,-5,-6,8,3,9,-6,2,1,6,-10,-6,10,8,9,3,6,-1,-6,-2,-10,8,-5,10,10,9,8,-2,-5,10,-6,-6,-1,-5,-7,6,7,3,-6,9,-8,-4,2,-7,-8,-1,-10,-7,5,9,-10,-1,-1,-10,-9,1,-5,-4,-2,-6,-7,-5,3,9,-7,-1,-3,3,-9,9,1,-5,-6,5,-1,-8,-7,7,-6,-9,-5,9,3,2,-4,9,-2,4,-5,-5,10,-5,6,-7,6,-7,-6,-9,-10,-2,1,-2,10,-10,-2,9,6,-9,4,7,3,5,2,-10,-4,-9,-8,9,-6,2,1,2,-5,-4,-8,-3,6,-10,-1,-1,7,-3,8,2,10,-3,-7,-10,5,-3,4,-4,3,9,8,-10,-7,-1,-4,10,-9,5,-6,-6,9,-6,5,-3,5,2,-1,8,3,4,2,-1,7,-7,-1,5,-4,2,10,8,1,-9,-2,10,-6,-5,8,-1,10,-7,-1,-10,3,4,1,-9,4,3,-2,-1,7,9,2,10,6,-4,10,-3,7,-6,-1,-1,10,-1,8,3,10,1,-1,-5,-8,5,-2,-1,-6,-7,2,2,-3,6,-6,7,2,7,10,-2,10,8,-8,9,8,2,4,-3,6,-8,-2,-8,4,-2,9,5,-3,-1,6,4,-6,5,10,-8,-1,1,-1,10,6,2,-5,-5,1,-9,6,2,-10,5,-7,6,-9,-3,-6,6,4,9,6,-8,-2], dtype = "uint16")#candidate|10611|(792,)|const|uint16
call_10609 = relay.TupleGetItem(func_5908_call(relay.reshape(const_10610.astype('uint16'), [11, 12, 1]), relay.reshape(const_10611.astype('uint16'), [11, 12, 6]), ), 2)
call_10612 = relay.TupleGetItem(func_5912_call(relay.reshape(const_10610.astype('uint16'), [11, 12, 1]), relay.reshape(const_10611.astype('uint16'), [11, 12, 6]), ), 2)
output = relay.Tuple([call_10596,call_10609,const_10610,const_10611,])
output2 = relay.Tuple([call_10597,call_10612,const_10610,const_10611,])
func_10624 = relay.Function([], output)
mod['func_10624'] = func_10624
mod = relay.transform.InferType()(mod)
mutated_mod['func_10624'] = func_10624
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10624_call = mutated_mod.get_global_var('func_10624')
call_10625 = func_10624_call()
output = call_10625
func_10626 = relay.Function([], output)
mutated_mod['func_10626'] = func_10626
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10353_call = mod.get_global_var('func_10353')
func_10354_call = mutated_mod.get_global_var('func_10354')
call_10684 = relay.TupleGetItem(func_10353_call(), 0)
call_10685 = relay.TupleGetItem(func_10354_call(), 0)
func_8866_call = mod.get_global_var('func_8866')
func_8868_call = mutated_mod.get_global_var('func_8868')
call_10694 = relay.TupleGetItem(func_8866_call(), 0)
call_10695 = relay.TupleGetItem(func_8868_call(), 0)
output = relay.Tuple([call_10684,call_10694,])
output2 = relay.Tuple([call_10685,call_10695,])
func_10703 = relay.Function([], output)
mod['func_10703'] = func_10703
mod = relay.transform.InferType()(mod)
mutated_mod['func_10703'] = func_10703
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10703_call = mutated_mod.get_global_var('func_10703')
call_10704 = func_10703_call()
output = call_10704
func_10705 = relay.Function([], output)
mutated_mod['func_10705'] = func_10705
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6366_call = mod.get_global_var('func_6366')
func_6368_call = mutated_mod.get_global_var('func_6368')
call_10779 = relay.TupleGetItem(func_6366_call(), 0)
call_10780 = relay.TupleGetItem(func_6368_call(), 0)
func_2016_call = mod.get_global_var('func_2016')
func_2019_call = mutated_mod.get_global_var('func_2019')
const_10791 = relay.const([-9.751320,-5.678038,8.488732,-4.251553,-6.468112,0.639865,2.781599,1.423494,-1.087427,0.716974,7.727343,6.016159,-0.541047,8.885448,-2.300214,-6.094430,-8.589829,-2.838253,-7.446435,-0.533259,-6.532262,4.342736,9.069598,-8.368887,8.228068,4.746721,-4.813646,-4.362644,7.387264,-0.155774,-9.112728,-0.553077,-4.004242,-5.316087,7.918481,7.505294,-2.848759,1.193293,2.059380,-8.412479,5.872005,1.815144,-2.524661,-1.539675,6.433203,2.154109,-0.255054,3.331881,-2.747554,3.833846,-8.260062,6.431929,6.409107,8.548027,6.392263,-1.039347,8.454236,-5.030797,3.581942,3.602161,5.956808,-8.610295,-5.556164,-8.225979,4.063142,3.040017,-3.763678,-0.609525,0.501354,-7.125483,0.691627,-2.272174,1.794808,-0.633299,9.422874,8.369106,1.801842,-0.884493,1.378795,-9.500135,5.293706,7.877503,7.958106,4.160814,7.226676,7.815418,-5.750044,1.390358,-5.696818,-1.443071,-5.453481,-9.516356,5.740928,7.961390,-2.517416,2.729774,3.003503,2.928834,0.531659,-1.210650,0.798264,4.001637,4.050483,6.779327,3.478769,6.351580,7.542029,0.663038,0.904702,-8.651909,-8.227060,3.929000,-8.898247,-1.865066,6.235500,-1.765231,-4.490787,0.431230,-9.664911,0.872237,-1.254212,5.858987,6.494721,0.352695,-8.515213,-7.524070,7.350902,-1.178728,-8.548373,5.915835,4.166689,-4.803514,-7.233526,2.744532,-3.211510,-0.955681,-4.490365,6.444663,-0.134003,-9.006555,-2.016858,6.288193,-3.541509,1.076191,7.194532,-9.656191,-8.061209,-8.466379,-1.368904,3.118379,9.813419,-0.189160,-7.528317,9.872843,-8.077192,3.702809,-0.417749,-1.320181,-0.383981,0.055019,2.001015,5.588266,7.651338,-6.955996,5.364298,-9.249307,0.072699,-5.808105,-2.025134,-5.658847,5.174526,0.146159,-8.752469,-6.439539,8.605497,-6.700107,-3.851980,-1.301531,-5.932493,5.152400,1.463730,-2.182182,-6.233297,-3.922081,-7.064805,6.207124,-0.283776,2.982814,4.598049,8.467796,1.247528,4.701632,4.237997,0.487929,-8.071582,-2.803850,3.968345,-4.975160,9.766482,-8.617364,8.061477,-0.890482,0.052388,-4.048236,0.150240,9.724057,-6.704255,-9.602526,2.304810,5.944663,8.916642,6.792171,-3.599756,-5.455160,-4.553742,6.311293,3.512012,1.499539,9.691279,2.424865,8.053231,2.013611,9.503432,-3.309092,-6.352956,-1.317880,2.193651,-0.049425,7.895706,3.391327,6.044714,8.731381,-8.006659,3.752160,-5.467809,9.312372,-2.942784,4.750208,-9.850332,-1.142403,5.153840,-4.073681,-2.402219,-4.755235,-3.486286,3.005764,-4.932994,-1.830229,6.991404,-9.534946,-5.806056,3.746250,2.943877,-0.463182,-1.869515,-7.980085,9.563224,-9.992538,3.339572,6.484994,9.018498,4.333741,4.785064,-4.767169,-7.961604,-7.120433,-1.961361,5.622260,2.881221,-8.034462,-3.478540,7.741814,1.633168,3.371358,7.750084,-1.310556,3.436706,-7.259937,9.604910,3.958683,-6.047069,-8.166794,7.355634,-1.285117,2.217974,4.422417,2.706938,4.499952,-6.220947,1.266762,2.889828,-7.272189,2.275229,5.810640,-2.617888,-7.493881,2.568582,9.113747,-1.074947,2.908750,-2.802613,9.896313,2.361190,2.285448,9.780711,1.785002,-4.816188,3.826460,2.033123,9.048753,9.813211,-3.755719,-5.641108,-0.536892,4.791868,0.556973,-4.376582,8.049065,3.379703,4.211671,0.031971,-1.675572,2.465834,8.887850,-3.966166,-3.329192,8.451754,-2.219329,-9.526635,-8.358340,-2.838360,-6.278021,2.407382,9.300953,-3.793324,8.815572,9.402752,3.018356,-7.111461,1.581008,7.596223,-5.587571,-4.723207,4.650497,6.793097,-7.113688,3.310224,-1.378437,-7.451797,5.120467,1.951465,-2.280801,-3.956470,1.952185,7.234642,-6.862904,-7.505901,3.543862,3.753216,1.084979,-5.686453,2.513871,-1.779984,-8.667898,0.546883,5.704533,6.419072,5.550949,-7.599556,-2.516864,-7.949443,-8.714480,3.069544,2.576935,-0.727128,-6.082041,9.885234,-0.291886,2.363255,-2.450576,5.255471,1.592236,-2.859180,-5.062326,2.446363,-1.719738,5.711821,-7.433834,-6.764798,-1.489767,6.977108,1.195687,-6.656104,-8.139102,4.647796,-2.138644,3.861797,8.283610,-2.699430,-0.502819,-5.833461,4.514377,9.561196,-9.527363,-9.438572,1.576556,6.446902,-4.606462,-2.207058,-6.710248,6.966271,-8.620373,-5.982745,-2.760721,-2.699042,5.713847,0.116756,-9.868324,-6.498179,-5.077376,0.304281,-3.798739,3.333597,-5.928319,-2.556123,3.065796,2.313624,-2.106721,2.762441,1.599495,8.805778,-1.891429,4.210225,8.975986,1.134963,-3.430558,2.122720,-0.375937,9.544457,9.260560,8.166893,-1.535015,-2.374428,-4.243574,2.099893,4.181398,-8.061015,-3.881122,0.087732,-5.038885,1.822293,5.610888,0.904551,7.224961,2.287384,-9.952969,4.605402,1.793753,7.272858,2.051634,1.278943,3.358815,-8.680852,-0.982218,-1.832672,2.340721,1.031404,-0.421004,-2.769243,-1.913768,-3.060923,-9.357596,7.702845,-2.880227,5.876682,3.095639,-2.774048,9.748169,-2.914876,-7.254719,-8.755758,1.992564,7.040171,-0.746862,1.934400,0.407522,-3.706555,-8.405721,9.877709,-4.829040,-6.036021,-6.712071,4.225652,-4.243325,1.462759,-5.312878,-6.328034,1.085118,5.359186,4.848242,-2.372560,6.000961,-3.742730,-3.038586,-5.273280,-4.557680,4.003635,3.101251,-5.785964,-9.034301,3.443448,7.863426,-2.457127,1.357167,0.699941,4.673010,2.172178,-5.844300,-4.396055,1.021094,6.956190,4.337796,-0.321598,7.360706,-8.646659,-0.758502,-9.159101,3.130261,4.737655,3.605892,6.196161,-0.871306,1.591295,-0.676938,-9.145093,-9.519621,-6.875435,-7.665334,-5.019659,8.159191,0.178750,-9.278927,-7.486302,9.125415,-4.302312,-1.845087,4.808985,-8.576549,2.737613,-9.397124,3.098015,7.913918,-9.897978,-7.210735,-3.154894,-3.087087,-7.737300,2.824180,-8.975654,-7.228698,6.788278,9.059429,3.544532,4.086660,-9.663327,-1.319401,-1.012786,4.406973,3.266754,-7.108554,-7.527875,-8.274237,9.467389,2.453874,-1.085932,5.980768,-6.930444,-3.800745,-4.187122,-4.522707,5.658858,-4.849450,-5.195207,-6.137448,-9.907150,9.074555,2.433674,9.043780,-3.871884,9.792082,-4.370548,-7.140430,3.732503,-8.278411,1.083227,-4.107027,-9.759850,-9.367424,7.695138,-8.952160,0.561734,-0.001931,1.427141,-2.352417,-1.055605,-5.448221,-5.620345,7.166883,0.908511,5.153460,-9.130326,6.691474,4.569844,7.415130,1.518784,-9.908609,-0.533039,1.976513,-5.001475,6.673877,7.798046,2.637573,-4.661123,5.691197,-6.076707,-4.306388,2.096552,9.284078,-6.157876,6.108125,8.324580,-4.544943,-4.223189,6.263862,5.258900,3.899370,5.333264,-0.460934,0.764048,9.641042,6.781993,-4.074682,-0.798676,8.529963,-4.954075,-9.418673,-9.101732,4.894379,-8.350468,2.336130,9.364593,3.084500,5.104957,8.959669,3.638545,-4.259054,-3.813596,8.019349,5.981093,0.372797,5.279172,9.646528,0.536817,-7.959837,7.139774,1.330055,-3.436611,5.450565,9.582486,-8.547166,-8.907818,7.552343,9.616489,8.196902,2.671265,8.742096,-8.788500,-9.002155,1.466899,0.214945,-9.264134,-2.461153,6.957635,-4.474225,-9.692473,-4.263905,3.377310,9.287117,-7.570943,3.283039,-9.455001,0.622187,4.056458,-6.887954,-7.342912,-5.964828,-8.629028,-2.921678,-1.226939,-1.992838,0.830473,2.748383,-3.089791,-3.356811,5.233637,-3.558761,-8.643527,-2.260671,9.112198,-0.764791,9.557244,-7.901342,7.397080,8.489877,7.320764,-8.038458,-4.967760,-7.272989,7.608503,-9.210902,-5.905574,-1.418982,-6.238481,8.135526,3.206994,0.640091,-3.683257,-1.018725,-4.294106,9.759486,-2.311277,0.635224,-3.481427,3.539429,-9.456626,-4.328779,1.191057,5.461719,-1.176224,5.959230,-0.533966,-3.194706,2.436027,2.341005,-5.009073,9.217984,-3.770874,-0.629365,-9.414638,-5.182816,5.407553,-8.595803,8.167484,-8.376868,-0.667292,1.301464,-6.669888,9.513441,4.590417,-0.667526,6.504908,-7.619049,-6.586767,-0.028432,-4.836065,7.835360,-1.723390,-8.978212,0.286805,5.161185,6.148255,-5.324283,-7.809385,3.421167,-4.503803,4.925983,7.857788,-1.969640,-9.838916,8.005026,4.651645,-0.040275,-1.903140,-0.343335,6.977618,8.123840,-5.451550,1.658162,2.459511,-6.837336,-5.716658,4.308465,9.611580,5.023844,7.604950,2.067999,2.183148,-2.348805,-4.733863,0.123174,-7.463674,-1.212344,-9.025717,2.666704,-2.207670,-8.773984,7.587126,0.666224,8.773807,1.915855,3.339306,3.635544,-4.761750,-9.470533,-1.110859,2.449892,-3.878186,-5.888935,6.385737,-0.910583,6.771980,-3.296541,-1.691574,-9.864679,-2.254243,-7.151387,-2.643628,-2.118941,5.835336,4.207727,-0.868661,-7.530860,6.623223,4.997013,-3.083340,-6.431328,-4.857958,4.293294,-5.120022,-9.281380,1.504424,-4.079530,2.204319,-1.218550,6.180660,-2.377488,-8.711734,-7.150237,3.043649,-0.232829,3.269760,-1.214698,0.599081,7.496953,1.473682,-1.469838,-6.531444,-4.713201,2.941182,6.903037,-6.619633,-3.788051,-7.723376,-5.160448,7.702179,-4.073610,-0.165309,-4.688693,-8.530013,7.141074,-2.374873,-3.791726,9.893366,-3.595270,3.433696,2.058072,2.011284,2.519733,4.942698,-3.637989,-8.853461,0.396491,2.698882,6.168875,3.773526,2.236485,-6.271059,0.619629,-2.412446,2.400127,2.294779,3.082822,-4.525375,-7.040218,8.790329,-2.655030,2.531414,-9.219750,-5.089657,-0.407151,-6.772510,0.116264,5.782912,-3.870841,-7.333026,-6.136877,4.001689,-0.154608,-9.289897,6.660818,2.785501,0.594885,1.720744,4.391578,6.863076,-3.782322,5.726423,5.311987,-3.970378,-5.206190,6.424075,-9.876060,1.881131,-7.446095,4.801065,4.987304,-6.754456,-4.577993,-5.381815,5.820143,6.602207,8.484909,2.058610,-6.987969,4.689281,-7.031144,7.704865,8.021818,2.888873,-8.013553,2.198053,-7.828622,-1.704458,3.748219,-9.353452,1.414129,-0.493565,0.436990,5.640041,3.147382,-7.316636,2.649954,-8.532855,-4.011152,-9.353562,3.803432,-2.211038,5.233366,-8.570101,2.216282,3.911153,-8.941306,-9.312368,6.317290,8.389158,-1.674370,-6.296120,6.583089,-2.098998,7.862923,-6.837964,-2.466303,-6.919665,-3.569069,-8.737660,0.301288,-9.313513,-6.650400,1.520637,-6.867587,-7.424492,0.624920,-3.953660,-4.065193,-0.476198,9.658659,-4.568447,-5.990791,-5.764215,6.678714,-9.312660,-0.367215,-9.318821,5.337973,3.755348,8.669161,-8.210163,-2.228012,9.529258,2.021022,-0.280243,4.407496,-6.760204,3.132619,-8.280570,-6.143247,-1.062891,2.532816,0.664887,-9.424106,-4.858184,-1.129202,-0.637388,5.152601,9.036608,-7.702717,1.212467,-3.990561,7.959990,-6.744868,-9.026411,-4.429520,-8.931542,-9.565635,9.112155,-0.517377,-8.526907,-7.885871,1.218470,-7.547176,1.723208,3.058064,-6.597902,-2.592020,7.279810,7.041784,5.493445,-5.270669,9.649617,6.118671,-7.164882,-2.858037,2.134458,9.325959,-3.209459,0.870844,-7.940451,4.413878,5.127717,-5.729614,-0.991913,-9.875751,-2.129801,5.067168,6.215083,-3.501837,-7.818525,-1.487171,8.299624,2.319881,-2.353371,0.825726,5.101469,1.787828,7.182542,4.550805,-1.233734,2.966725,0.809073,-7.718529,-6.458096,-3.463747,-7.000409,-7.396507,-5.403204,7.075111,-0.120670,-4.114406,0.003685,-0.492879,0.584833,3.561062,-8.943043,-6.689201,1.906630,-9.067684,7.242105,8.354803,-2.264681,-9.632587,5.854486,5.217166,-4.063544,3.821961,5.634631,-3.112537,6.518668,-2.551246,4.523750,6.053963,-2.820492,-9.130516,-7.904542,4.772011,4.797469,-5.621874,4.889897,-5.438141,2.389340,-7.587964,-2.958953,1.762595,-6.348546,0.033645,0.717342,-6.243546,-3.745128,7.286085,-2.174225,3.310899,-3.787000,-3.165835,6.859386,-5.758104,-8.469855,-5.218962,2.106700,1.420788,8.411307,1.390101,6.667849,-9.754377,-6.122127,4.235631,-1.201901,-7.803343,-4.111922,5.604132,-3.870245,-1.490031,-5.605797,0.058174,-1.988612,2.435713,9.451808,-9.387057,-4.889669,-8.732815,5.749928,-7.018320,1.454831,-6.693174,8.039450,2.512816,0.788664,5.303290,-4.210140,-4.636394,5.418006,4.260510,-7.355116,8.865980,-1.501979,-4.658268,0.682009,-9.153340,9.248280,-6.539688,-5.836051,-9.677359,2.487580,3.571509,-1.029137,5.320201,4.766843,-6.054008,-0.245118,-3.548869,-8.308792,-3.070463,8.876813,-1.154677,-7.768838,-1.689590,-1.451462,-2.787834,-9.788759,-9.615351,7.725309,-2.997155,-3.002285,9.294673,-9.465310,0.886575,-9.863500,3.469505,-1.651020,-5.351648,-0.112019,4.295896,-8.429973,6.954835,0.006187,9.301260,1.461126,6.882030,7.993817,-7.675334,8.489189,-3.769672,-2.605962,3.498677,5.963455,0.470017,-9.674075,-9.372745,-0.420790,5.594585,7.935981,0.875149,-1.688251,-9.285578,5.188661,-8.500170,7.415255,-0.848400,-7.700714,-2.591804,7.250265,2.135806,6.823945,-7.661048,8.892882,1.691247,1.843732,7.257365,-9.456716,2.854005,4.626591,-3.826971,-7.298070,-2.523654,-1.524173,-9.327016,-2.214585,-4.859781,9.312862,6.228289,-2.715685,-7.280554,9.708672,-8.102426,1.742547,9.043916,7.665939,9.053188,-6.898977,-6.926245,9.661210,2.106354,-4.183681,1.177468,-5.844733,3.949349,8.820986,-6.955351,7.577839,-2.518730,-6.254066,7.133042,7.856980,2.022095,-2.737738,4.009907,-0.321373,1.022532,-2.597921,3.655906,-0.123564,9.575987,-6.452861,-4.310353,-7.119008,-8.996955,5.297304,-8.620277,1.274094,-4.346276,3.783602,7.936677,-9.407973,2.159745,0.778145,-0.456218,3.357335,7.722590,9.680021,1.271598,5.806025,8.162893,-0.095162,2.527283,5.867365,5.604724,2.688024,9.763569,-4.583403,-0.220553,3.720964,5.925363,9.236390,-6.942544,-9.387659,-6.900773,5.665704,-8.482097,-9.941315,7.763984,-4.601140,-7.974953,9.311406,4.739409,-5.647985,2.088359,-6.341308,-9.455658,-4.954615,2.518404,2.072709,-0.481753,6.158409,-1.262367,-8.169456,5.523091,-8.490987,2.725046,3.978016,4.933305,-4.936388,7.954107,9.399622,-7.194248,-4.466693,7.805811,-8.238726,2.233345,-6.130107,5.352392,2.596055,8.919761,-1.519554,-9.954925,-6.335850,-3.750357,-5.117374,7.497553,3.368772,8.993628,8.593092,8.109857,-6.071680,2.236781,-2.920996,-3.279317,-0.692589,-3.523712,-2.866004,-5.811045,-8.720568,1.366193,9.134670,5.828370,7.344125,-9.629806,4.831019,-7.164465,-0.836567,-7.876719,-0.940885,6.455854,9.438169,6.442459,-7.936055,-0.886192,-8.859932,6.554066,-5.473275,2.992688,-0.015250,1.534084,4.957312,6.194381,0.615873,1.782133,-0.888354,-9.504709,-1.982132,1.656389,-0.506446,5.287514,-3.125740,-3.841297,4.724249,-3.698234,7.466763,4.497800,1.919595,9.551669,8.601110,-1.453028,1.052380,5.579731,-8.706096,-8.444296,-2.335227,-0.268165,-0.310966,3.836392,-3.461320,8.078487,5.456293,-6.861423,-1.727481,-6.707423,-6.902704,2.313275,3.086161,4.131413,-7.307435,-3.976443,-3.115347,7.668698,0.671379,-0.428621,-0.394185,-3.369466,0.061005,-7.583399,-0.332389,9.779081,8.712757,-1.560280,5.481875,-1.829848,2.524879,-4.258636,5.773474,-7.235645,-8.069447,-4.765087,7.156269,6.845118,-9.035025,-8.908905,-3.545120,3.894332,-5.158234,-5.397316,-9.787223,4.186698,-4.656439,6.251301,3.807499,-8.591130,-9.721341,-4.182032,3.409642,-6.514456,7.997753,-6.534708,-8.619680,6.777340,5.043950,-2.416165,9.052805,7.126348,8.240240,7.530626,3.048066,0.073785,9.733876,7.574127,9.056125,-0.876397,-9.921311,-6.236453,-3.631315,0.455638,3.715232,5.797106,-9.860637,-5.084734,9.446115,-2.287479,-6.493754,5.739476,-1.423765,-0.943342,-9.491617,-7.391031,-5.338870,-6.651663,8.195306,-4.330859,5.287515,8.828121,3.873437,-0.783742,7.289084,5.748387,-0.293751,2.089963,3.848281,-8.396052,-8.810912,-9.855130,6.502271,-5.690091,-2.570196,8.542353,-2.638096,2.479507,1.542311,4.787937,5.670636,-4.389403,1.314725,-5.038961,6.092662,-2.094774,7.096945,-7.075527,1.423682,3.504439,-7.676661,-7.149127,3.594053,-2.356043,5.689462,-7.348865,-8.536923,-1.780281,-7.094043,-7.046370,8.775127,-3.577515,7.290793,3.484834,5.115402,-1.835622,4.516253,-0.576998,5.917752,6.816524,-8.771223,4.996600,-1.401044,-1.724158,-6.470702,3.815961,-5.078888,-0.116322,-0.456694,-4.886699,7.914146,3.897516,8.017535,6.641457,-0.842972,9.498424,-3.774640,5.354655,-8.690859,-8.969452,-2.711288,-8.928898,9.886384,-5.853675,-1.295227,1.684047,-0.064072,0.507737,-8.379120,-3.138596,9.316764,-3.342849,-1.795254,1.133271,4.745250,2.694128,0.447839,-9.551362,-0.356054,-7.312694,-1.528161,-3.969624,6.079800,-3.829724,3.362373,-4.994247,4.807416,-2.962341,5.790220,-9.032297,1.618256,-2.301977,-0.487644,7.250395,-3.939530,4.699783,3.887044,7.908332,5.256573,6.287198,-3.189013,-5.696066,-7.320999,8.742622,-3.318494,-1.119036,6.410977,-9.812116,-1.013896,-3.030524,9.205009,1.430708,-3.950510,8.328506,-2.689780,-9.692471,0.691620,-6.770869,6.715209,4.323234,-5.166628,-2.474637,-3.261622,3.660310,-1.324540,3.271708,7.937907,0.759897,2.939400,-5.933897,6.581348,-4.605391,6.257627,-2.523022,-5.403744,-3.746351,6.639568,-1.539197,1.508219,-8.877550,-9.922592,1.806931,-5.382520,4.714175,-1.223059,-6.057805,-0.513086,-6.283977,7.297437,-2.604177,-1.193790,-0.924921,2.496738,5.152167,-4.258893,3.287300,-2.055320,5.697091,6.030519,-9.466853,-3.236674,9.914689,-6.435739,-8.504902,-2.639115,4.231882,1.328358,-8.573300,4.043925,2.834065,-1.627300,-8.757651,1.985007,2.120302,-6.918310,-8.938679,-4.417944,6.764941,7.726501,4.477790,-8.166560,-2.824787,-2.523034,1.446871,9.468364,-4.532868,6.346357,-3.229172,-1.887229,3.886737,2.388102,5.384675,6.066318,-8.320136,-0.412955,-3.163342,3.345393,-6.169231,3.505882,3.235304,-2.496796,-0.426678,-2.211525,-4.434887,6.799390,5.774114,-5.370427,-5.832792,5.157244,-0.910058,-7.863566,-7.298269,-4.891667,-6.359450,-3.905690,-4.091753,4.374413,-9.391667,0.571872,-3.088882,-8.316833,4.282376,-1.902146,-1.365580,7.445872,4.995437,2.083983,3.080338,-2.185302,-6.259389,1.869696,6.132155,8.966582,-6.388351,-8.780928,7.255553,-0.051884,8.834923,4.202024,-6.918481,2.761877,-8.329588,9.168522,-6.863573,1.444761,1.419184,0.042553,1.446805,-5.888413,3.583707,-5.888466,-3.828058,8.800085,2.978262,5.363769,6.525134,1.874199,-1.561026,3.500809,-1.394730,8.869860,0.381248,-5.105442,5.205973,-3.068534,-5.378121,-5.975216,9.381038,-6.635582,-0.130717,-3.713885,-9.576788,9.448853,7.043118,2.320999,-9.092112,2.606277,4.902108,-4.525909,-7.384216,4.600080,-6.979724,4.305303,-5.292516,-0.498412,-8.817896,-2.491527,8.308531,4.233457,9.874226,-8.076094,3.492970,0.888987,-8.806668,0.613453,9.761635,7.348695,-3.262943,1.588765,-7.680142,5.981961,3.417232,-3.917638,-1.704254,4.225740,-2.950698,3.749762,5.530711,1.065832,-4.580429,-7.390006,6.555182,6.078492,8.469337,-5.653908,4.377203,-1.282208,-1.940079,3.500351,-8.637588,-1.642422,-1.321589,-5.251615,-5.968554,0.730519,-1.366518,-1.057537,0.894369,-3.945117,5.194346,1.060444,-7.495678,5.187995,-6.536740,9.500255,7.353221,6.411522,7.749276,0.975406,4.183117,7.292632,-8.603070,-0.279441,1.176087,-7.812708,3.723679,2.676965,2.403786,-9.268112,2.568259,-1.989361,2.269668,2.775182,7.529076,3.835278,5.391554,5.864636,-0.392840,-5.525478,-0.069690,-1.032937,-2.758362,5.215473,-5.535131,8.635877,-3.770542,-4.899879,8.363756,-1.154110,-9.636662,-5.677440,-6.941109,-5.347196,4.070560,3.711944,0.641382,9.207009,7.393574,-7.840709,3.102551,-0.507217,3.463107,2.180406,-1.508280,-8.063551,7.309239,-5.148479,-3.525691,2.823619,0.017826,8.623082,-0.373993,-7.827296,7.271497,-7.541976,-4.737008,-3.441212,7.108315,6.258838,-4.795221,9.571334,-4.982303,2.765655,-3.661093,-3.919825,6.754098,-5.433144,-6.150882,-5.181681,5.421861,-3.909660,2.279056,-6.193765,3.435024,-2.365980,4.955171,6.001268,7.124095,-0.956263,4.008199,-4.026667,-5.141298,4.887792,1.298601,-7.491753,-7.996267,9.365156,-8.449984,-2.623342,3.114714,4.118758,-3.430258,-1.186327,2.256228,3.312118,0.318584,-6.401476,1.969726,-5.041671,4.941542,3.522273,-6.654699,1.192907,-0.063169,-4.758915,-6.805904,-0.688606,-8.357074,-1.984291,3.098793,6.086839,0.608600,-3.516558,5.659949,-8.656125,-2.382791,-2.260735,-2.395377,8.354023,8.831269,-5.363956,1.292606,2.535709,0.759942,8.124601,3.605597], dtype = "float32")#candidate|10791|(1980,)|const|float32
call_10790 = relay.TupleGetItem(func_2016_call(relay.reshape(const_10791.astype('float32'), [11, 12, 15])), 3)
call_10792 = relay.TupleGetItem(func_2019_call(relay.reshape(const_10791.astype('float32'), [11, 12, 15])), 3)
func_6666_call = mod.get_global_var('func_6666')
func_6668_call = mutated_mod.get_global_var('func_6668')
call_10795 = relay.TupleGetItem(func_6666_call(), 0)
call_10796 = relay.TupleGetItem(func_6668_call(), 0)
output = relay.Tuple([call_10779,call_10790,const_10791,call_10795,])
output2 = relay.Tuple([call_10780,call_10792,const_10791,call_10796,])
func_10804 = relay.Function([], output)
mod['func_10804'] = func_10804
mod = relay.transform.InferType()(mod)
mutated_mod['func_10804'] = func_10804
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10804_call = mutated_mod.get_global_var('func_10804')
call_10805 = func_10804_call()
output = call_10805
func_10806 = relay.Function([], output)
mutated_mod['func_10806'] = func_10806
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8752_call = mod.get_global_var('func_8752')
func_8753_call = mutated_mod.get_global_var('func_8753')
call_10922 = func_8752_call()
call_10923 = func_8752_call()
func_9854_call = mod.get_global_var('func_9854')
func_9857_call = mutated_mod.get_global_var('func_9857')
const_10927 = relay.const([6.722962,-6.062680,-6.644194,4.115076,8.748160,9.326197,-4.571435,6.593728,9.192889,5.607857,-7.708857,5.249585,-7.907509,0.666780,7.872524,-0.741957,-0.640043,-4.886827,-3.790922,-0.207280,5.295820,1.738424,6.409604,-3.191948,2.204511,-8.010785,6.099825,-5.969878,-5.366084,2.836938,-2.250488,-1.090285,-7.781288,9.677185,-0.518928,-8.533656,-9.362387,7.475107,-0.878202,-8.418799,5.691503,0.251298,-9.662225,-3.807082,-8.834099,-3.585335,-5.636224,7.246156,-0.407140,4.466584,-6.050275,-2.196375,2.664977,-0.095510,2.161166,2.033648,-8.433124,-2.329895,4.594270,8.002865,-1.016257,-8.265057,-8.724835,-6.549139,-3.938654,1.545662,5.756788,3.721699,3.327943,8.171051,5.310194,6.742735,0.838758,-6.618363,3.517755,-8.777861,-2.317289,-9.730431,1.300948,5.590768,-6.195211,-7.470973,-3.201773,1.502441,-2.309565,3.520375,6.877101,1.873630,-2.201665,8.354788,8.427213,9.553688,-6.950348,-5.703181,-1.956237,-0.812936,-6.686846,2.683223,-5.367885,-6.190556,3.095683,-3.959091,-9.494844,2.202140,-4.627704,-1.038823,2.855947,8.930743,4.346611,6.237643,4.245592,0.157942,-3.541323,-3.755006,2.094438,9.438604,-1.437495,7.121230,2.382399,3.394156,-9.493610,4.669913,-2.850213,2.537107,-1.671054,0.702550,-5.484672,8.644492,5.939606,-0.432716,-6.166562,2.688502,9.552173,-5.345920,-7.663968,-2.029100,0.208397,-9.321989,-9.784136,-7.561247,-8.810627,-9.469380,-1.672245,3.600226,-5.007812,-3.191985,-3.929930,5.588732,-3.370870,-2.867021,-4.326560,2.537106,7.297871,3.237736,-7.412440,-8.461481,9.751226,6.660090,1.182025,-8.144061,-1.244156,-5.782658,-3.924889,5.130292,2.780080,6.628120,-6.836268,-6.156977,-0.535271,-4.243692,1.370314,7.200878,3.603903,-2.797516,-0.767096,-9.534547,4.506813,3.637020,8.391903,-9.658745,-6.634364,-7.511000,0.345248,-1.877698,-4.099343,-0.280246,-4.412235,-7.614261,-1.357983,-8.959364,4.283419,-8.543925,3.257818,6.115292,-7.845120,-4.599453,3.751054,5.481712,1.456877,8.897384,-5.462699,2.833087,3.682163,9.661163,-6.253690,-0.794039,-6.447658,-6.480042,-0.142142,-5.545298,9.020184,-5.131487,2.602034,5.179142,8.118631,-5.265046,-4.505629,-9.330128,-4.814250,-0.092881,2.176295,-6.095051,9.482948,-5.874651,-4.843537,-1.622607,8.248573,5.803234,-7.615505,9.854778,3.254564,-3.966828,2.036453,9.906284,-7.582061,-5.917610,1.036319,8.024783,4.484439,-5.823806,2.897393,7.750030,8.942519,7.760245,-1.148275,-3.849843,0.252886,6.060186,5.597725,-2.320197,-7.076263,-7.137487,2.731415,-2.815249,1.346822,6.206881,9.062252,6.540928,9.755095,1.434019,4.360536,7.186824,-8.722029,4.766013,9.042897,2.717954,-9.030949,3.684739,5.399275,-3.869508,6.927492,1.992195,-0.452238,8.038318,-8.127906,1.210068,-5.565479,-1.964264,-1.413209,6.977460,0.703933,2.680320,-1.184901,3.759881,6.710462,-8.073255,0.163999,5.398456,-2.187268,2.509613,7.847485,-1.164572,7.685386,-1.993151,-3.253373,-0.606851,-4.746699,8.980883,-0.066126,8.357333,-9.682817,-6.806277,7.483141,1.166947,9.760121,-1.020835,-9.565490,-8.951528,9.799711,7.509076,-1.179775,9.242427,5.263585,9.687641,-1.020499,6.024411,-3.435321,-3.985633,-8.234640,6.220539,4.706043,-7.434866,-6.005872,6.349337,-2.533047,7.826022,-8.837777,8.588608,8.009600,-0.244848,2.344662,-3.992385,1.469719,-4.442671,3.755222,2.899717,6.038913,-3.051821,7.676318,-5.693051,-4.189333,-0.912069,-5.823509,-0.568231,2.653827,6.155236,1.038743,-0.568735,0.587395,-6.559995,3.642063,6.145645,4.014748,-0.234188,7.296144,-1.958208,4.312148,9.211192,1.893953,0.065438,-9.104224,8.575585,4.050766,-1.404056,-7.423720,1.730146,9.400058,0.198079,-6.793325,9.865469,-5.870632,9.350450,-0.372581,1.516280,-7.097559,-1.016657,-0.761191,-5.187309,-2.731310,-6.694043,9.925220,-8.868260,-8.385775,-2.360223,8.858951,-4.827408,5.250723,0.640486,-2.158803,3.126811,-9.815409,9.168007,-7.535642,-0.444131,7.099124,2.609341,-1.252497,7.276175,4.050143,1.294618,1.923132,-9.884528,6.024106,-2.086855,4.092642,-3.867175,-3.841947,-3.485438,-7.135022,0.366026,0.784455,-9.398857,-5.661151,-5.411779,8.506969,-7.208661,1.152441,-2.359218,6.402775,0.916536,-9.241883,-8.112062,5.201744,7.217272,4.917215,2.978410,-8.717883,-1.914659,7.006030,-8.339035,-2.411118,7.510938,-4.588732,-1.469333,3.093809,-1.891328,5.197750,-7.118446,5.444081,-7.395559,-0.369172,5.591786,-4.559049,6.365399,-6.771860,-6.618401,-9.186420,-8.434019,3.073078,-5.833707,-2.502049,2.827314,0.774782,2.447437,1.408893,8.016356,3.480287,-2.932694,-1.635404,-9.678632,-8.311992,2.949865,-6.442251,1.294480,-5.109123,-6.147726,-9.733543,8.839721,0.117181,-2.360578,-3.175518,-6.151094,-5.983892,3.479238,-4.444459,5.954038,9.583492,-7.277475,-3.598883,1.792418,6.808048,-9.489311,2.814196,-3.395678,-1.044156,-9.730496,-4.831197,-6.798062,5.719887,-5.242811,2.643688,-3.376348,-7.412952,1.014065,-8.495391,-7.332839,-0.948804,9.943768,-2.140480,1.582193,-7.963899,5.420554,-7.256273,2.127767,-2.627182,-3.220038,2.967270,-4.851297,7.345657,4.410454,-4.619620,5.215112,-5.455328,8.823962,-3.112099,8.610557,-2.492843,-7.932222,-4.612019,-9.826203,1.811513,1.727362,-1.423378,-5.595924,6.861280,0.692857,7.429435,-4.748510,-5.547878,-0.821284,-3.645881,5.194975,1.957722,-9.901119,-2.233213,-3.497183,1.535983,-2.043517,5.038784,3.385416,-6.480531,-4.982569,7.302306,4.692997,-2.573593,8.432266,7.265639,1.532270,-3.181313,9.113439,1.547080,-1.618599,0.738645,-3.712476,7.661761,6.974148,-3.529105,-4.353335,-6.846409,6.858929,6.616449,-1.385775,-5.967407,4.356809,8.534132,8.262512,-1.566608,8.181978,-5.757861,-9.503631,0.335391,-5.504398,-6.770703,2.876305,-3.168615,1.130451,7.924409,3.190183,-3.465779,-3.648071,9.777506,-6.179884,0.054651,2.631222,4.619977,-9.254852,4.917063,-1.812618,4.275216,-9.411837,3.133090,6.723612,7.356672,-5.214484,-0.009098,6.370932,-2.096802,6.326703,1.953711,1.481353,3.260618,6.383487,-1.021312,-5.207683,-2.425072,-6.050632,-2.949330,-2.424130,-9.864177,4.402116,-8.623267,-1.440262,-2.622052,-7.407280,-9.502751,-2.009883,8.042777,-5.633294,-0.819206,9.574142,1.283463,-5.855016,-5.226060,-2.351919,6.371915,7.347267,-8.310173,-3.060057,7.252528,7.988820,-9.930679,1.777172,5.518357,-0.955651,-3.344725,-9.911105,8.540882,-7.623450,-3.823611,-8.618411,-2.917834,5.374699,2.608095,-0.597444,-4.377920,-0.454367,7.883516,5.876547,5.112698,-4.211634,3.715738,7.826134,5.613846,-3.158493,-6.444815,2.337670,-0.530144,-1.427213,-7.994484,2.633669], dtype = "float64")#candidate|10927|(660,)|const|float64
call_10926 = relay.TupleGetItem(func_9854_call(relay.reshape(const_10927.astype('float64'), [660,])), 2)
call_10928 = relay.TupleGetItem(func_9857_call(relay.reshape(const_10927.astype('float64'), [660,])), 2)
func_9854_call = mod.get_global_var('func_9854')
func_9857_call = mutated_mod.get_global_var('func_9857')
call_10953 = relay.TupleGetItem(func_9854_call(relay.reshape(const_10927.astype('float64'), [660,])), 4)
call_10954 = relay.TupleGetItem(func_9857_call(relay.reshape(const_10927.astype('float64'), [660,])), 4)
func_6631_call = mod.get_global_var('func_6631')
func_6633_call = mutated_mod.get_global_var('func_6633')
const_10962 = relay.const([-3.921311,9.579792,6.908668,3.907859,-7.512679,-0.711210,-6.081902,4.491378,-1.030920,-4.750550,-8.914767,-6.265954,-2.717747,1.417802,8.621048,1.873021,-2.867308,-3.654418,0.315434,7.843349,4.849940,3.471429,-4.097455,-3.172408,9.251516,0.675259,5.454938,-4.714297,5.744171,3.572810,1.451865,-9.396433,5.797576,-7.767528,-5.240909,8.815543,3.019589,0.463206,9.510987,-1.567062,4.522410,-9.110151,4.873849,-8.851841,0.691764,-7.896619,8.797590,-1.418465,8.870764,5.938347,-1.350248,-2.359997,9.066545,1.243605,6.067822,-3.600565,-0.486216,-1.641833,7.842179,-3.410917,-6.426210,5.072451,0.056219,2.486169,-3.589515,1.200023,7.509907,0.741152,4.228362,1.154121,-6.116656,9.795352,1.797336,3.372304,7.153496,9.246880,-6.064243,0.335043,-2.950302,-8.060246,5.904391,3.304655,8.901558,-6.681316,2.920157,-6.073256,9.741254,-9.632451,9.671399,4.111989,7.087481,-4.145266,0.925053,-6.413396,-9.074690,6.140342,-2.168458,-3.613644,8.111929,-8.617793,8.006939,-7.014017,0.952950,9.816350,-9.785667,4.647910,9.210174,3.151405,8.684301,-8.766027,4.482654,3.321278,-5.523421,2.903601,-4.872592,-9.634045,-8.969948,-1.458753,-9.654047,-1.082882,-2.367443,-2.024885,-9.679672,3.475361,6.611484,7.307875,3.933217,-7.924088,2.815443,1.844294,-8.269820,5.139319,-3.710352,-4.043667,4.092950,-3.829723,-3.791552,3.387832,-2.380544,-5.788080,-2.029332,-8.380654,-9.400473,8.946716,9.947557,-6.084151,-8.511907,-4.461136,7.568110,7.564902,-3.277732,6.271873,7.383612,0.123714,7.173010,2.613627,8.100924,3.227939,7.159792,0.441479,8.241336,-7.892761,5.454755,-5.249049,4.947087,3.073420,8.558044,-2.807228,2.802938,7.057760,0.863720,-4.442850,-7.973903,-2.560779,-3.171652,-0.365871,4.023392,-0.833770,-1.684309,-1.362540,-1.890161,0.548400,-4.740145,-9.083609,3.227333,-5.280251,9.681634,6.857484,-3.687936,4.224129,-8.410675,-0.913193,-6.770316,-6.060908,2.668456,-0.009899,0.511055,-5.750197,5.097090,9.445874,-7.703347,6.212375,8.760512,5.144867,-3.895657,-3.813525,3.561028,7.989004,2.705431,-0.225694,-7.817285,7.376093,-1.438822,2.385372,-9.471235,-8.583069,-7.451686,-8.573090,-4.300475,4.500806,-1.578524,5.338383,-5.846795,5.493495,4.912483,-8.983714,-0.950995,9.621491,7.733891,-6.346504,2.897754,-8.732022,-4.225244,3.084904,8.973243,-7.317991,5.439836,3.715463,-2.774625,3.544318,-8.465396,8.878565,4.816764,5.621692,6.187587,0.794041,-8.737438,-2.690958,-0.400160,4.729135,-1.685070,-0.394257,-8.618815,0.825709,-4.935982,-3.080644,-4.824258,-9.392948,-5.965055,8.480409,9.348721,-6.587781,-2.953824,7.148512,4.144664,-1.098379,-5.922430,4.364618,-2.393936,6.501820,0.211774,7.416497,6.174083,-3.647263,6.814773,1.737764,-6.734111,-3.240832,-4.529753,4.793492,9.523723,5.813823,-6.253323,-6.111798,0.444141,-2.640478,6.402399,-3.414561,1.091857,-0.226117,-8.658320,9.582337,6.800943,-0.368722,3.747177,1.924297,1.941968,-0.070875,2.598533,2.472052,-4.729574,-6.471009,8.722835,-3.784718,4.165638,-2.332153,-6.798259,-0.997477,3.427131,5.378877,-8.289300,6.797896,0.537290,-3.517564,-7.617063,3.054398,5.094339,-6.414911,1.864640,3.361635,6.078560,-0.834208,4.450675,-7.015986,4.946853,-0.920372,-4.764044,-8.036778,6.454512,4.384224,-0.676151,5.313935,-9.678053,-5.993806,-8.537248,-1.824033,1.201417,-2.640110,9.650352,-4.321562,-1.202006,-7.192239,-1.570852,6.050022,8.905447,-9.862511,1.135392,9.759980,4.196753,1.935107,6.618483,-1.348540,2.517140,-1.332251,9.002837,9.888866,9.494941,-0.910491,-1.643460,3.568073,-2.526984,9.795706,2.691110,7.340742,6.528207,5.168914,-8.641465,3.779943,-5.717840,-6.165028,-0.577684,8.649983,-0.789530,-1.092050,7.359345,-1.770504,0.298171,-5.664970,5.628184,5.628489,-9.827296,3.180272,-1.410066,-4.051701,-6.686039,-9.698903,2.278633,-7.032346,-2.271346,-0.986845,9.398350,2.171493,1.324326,9.704698,-8.811810,1.554568,1.194979,1.649567,5.242824,-6.978153,9.726603,4.523586,5.825447,-2.904739,-8.201229,4.853966,2.675320,-2.198385,-5.046121,6.811520,-1.634773,-4.575398,-3.459922,-5.186425,9.488593,7.276829,1.845840,1.591306,7.697526,-3.602478,7.077126,4.798221,8.254052,5.032247,7.951484,-3.118376,4.797555,-3.848558,-9.346080,5.648357,8.430546,-4.538944,9.458959,0.712316,4.866133,-5.674637,-8.613197,-1.419702,9.386402,3.472941,-1.249370,9.852584,-8.730597,3.555970,-0.127782,3.549940,-8.167545,-4.852653,3.023924,-4.770019,-7.979022,4.522299,-3.941211,-3.427149,1.910059,-1.749472,9.045452,4.560900,2.205259,-7.299249,-4.027063,-2.094265,-2.190532,-0.745567,-1.339954,-7.157430,-8.106255,1.685447,7.100918,-0.921770,-4.860924,8.246918,-1.854140,9.660744,9.637058,3.596688,2.680177,-1.252069,3.656362,-0.487062,3.149436,-7.964822,-1.138937,-9.347235,7.704867,4.235008,-9.162183,8.144770,4.332917,-1.491182,-3.154984,-4.600260,6.028558,1.972450,-4.703625,1.969978,0.327739,6.106431,5.225141,0.075823,-0.864621,-7.354540,1.065778,-6.012007,9.604587,-1.902029,-0.736053,8.837817,6.817059,-0.625532,-4.647284,-3.337504,1.939121,-5.408583,4.179620,7.187020,-2.450288,-7.571179,0.254532,-0.674135,-1.716648,9.307094,-1.297388,-9.331657,0.282655], dtype = "float64")#candidate|10962|(525,)|const|float64
call_10961 = relay.TupleGetItem(func_6631_call(relay.reshape(const_10962.astype('float64'), [35, 15])), 1)
call_10963 = relay.TupleGetItem(func_6633_call(relay.reshape(const_10962.astype('float64'), [35, 15])), 1)
func_6968_call = mod.get_global_var('func_6968')
func_6971_call = mutated_mod.get_global_var('func_6971')
var_10972 = relay.var("var_10972", dtype = "float32", shape = (990,))#candidate|10972|(990,)|var|float32
call_10971 = relay.TupleGetItem(func_6968_call(relay.reshape(var_10972.astype('float32'), [6, 15, 11]), relay.reshape(call_10926.astype('float64'), [4, 16]), ), 2)
call_10973 = relay.TupleGetItem(func_6971_call(relay.reshape(var_10972.astype('float32'), [6, 15, 11]), relay.reshape(call_10926.astype('float64'), [4, 16]), ), 2)
output = relay.Tuple([call_10922,call_10926,const_10927,call_10953,call_10961,const_10962,call_10971,var_10972,])
output2 = relay.Tuple([call_10923,call_10928,const_10927,call_10954,call_10963,const_10962,call_10973,var_10972,])
func_10974 = relay.Function([var_10972,], output)
mod['func_10974'] = func_10974
mod = relay.transform.InferType()(mod)
mutated_mod['func_10974'] = func_10974
mutated_mod = relay.transform.InferType()(mutated_mod)
var_10975 = relay.var("var_10975", dtype = "float32", shape = (990,))#candidate|10975|(990,)|var|float32
func_10974_call = mutated_mod.get_global_var('func_10974')
call_10976 = func_10974_call(var_10975)
output = call_10976
func_10977 = relay.Function([var_10975], output)
mutated_mod['func_10977'] = func_10977
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6517_call = mod.get_global_var('func_6517')
func_6519_call = mutated_mod.get_global_var('func_6519')
call_10999 = relay.TupleGetItem(func_6517_call(), 0)
call_11000 = relay.TupleGetItem(func_6519_call(), 0)
func_10546_call = mod.get_global_var('func_10546')
func_10548_call = mutated_mod.get_global_var('func_10548')
const_11009 = relay.const([[-0.940939,3.473210,9.507294,-6.246364,3.179791,2.736478,-3.740717,-7.732692,-4.205577,2.887509,6.711540,-1.293125,-0.991951,-9.030490,-4.191526,-0.687861,7.425591,-8.705481,4.244632,-9.588820,-2.740308,6.376748,8.404414,0.735307,8.452326,-7.170405,-6.348570,-2.628923,-6.514555,5.965633,3.917569,2.497350,8.243331,0.532214,-7.928495,8.076403,-4.886152,-2.263501,2.904549,7.716919,-4.831464,4.115266,-1.860449,2.936320,-2.529903,8.344808,-4.891464,-4.501051,1.391737,7.399046,9.042154,-2.982895,7.549864,3.269849,5.028495,8.970027,-5.905258,-9.884043,-4.948016,5.486110,-2.830963,-9.681545,5.009095,0.241383,9.927989,5.507223,2.879045,-6.281633,-6.386723,3.056713,-9.298917,2.578225,0.502363,4.312442,8.999091,6.594690,5.891513,1.941509,-1.260789,9.732725,6.768791,1.116157,4.514855,3.851282,3.373101,2.342745,5.028724,8.339623,-2.074144,-8.784072,1.837019,5.133632,5.019221,-5.520491,4.534074,5.445218,-8.872745,-3.013763,-6.692675,7.912856,-1.043126,5.905308,9.585276,7.077344,5.103956,3.177972,9.616838,-9.284879,1.670018,-9.481515,-0.328176,9.301346,-8.336813,-6.337359,3.079869,4.437185,-6.914989,2.144220,1.838394,7.158832,-3.816680,6.734355,-9.229574,8.196862,-3.430186,-3.392782,5.167761,8.412822,4.611933,8.372427,4.882574,1.783616,6.079175,9.576863,0.769254,-7.936019,-3.632525,7.467300,2.606738,-7.193831,-6.641426,-6.185453,7.396983,-3.629320,-4.501429,5.390667,-6.423066,4.069611,-5.577217,-5.754196,-7.668282,2.504228,-7.447049,4.293341,9.052489,-8.401638,-9.843169,-5.816977,-9.433318,8.622252,-3.058330,7.379470,3.705906,-3.066522,0.173670,2.445897,0.612813,-4.243103,4.700023,7.598200,-7.959956,-5.897590,0.903374,-7.925988,8.327912,-1.311724,-8.916020,-9.180175,-0.228047,6.315423,7.349521,5.562251,-4.570676,1.370984,1.955589,-7.360118,-4.417137,4.374557,1.402932,4.556693,3.230610,-2.283401,9.285215,6.847740,-8.455009,-4.328341,9.063163,-0.751327,1.417090,8.077880,4.289914,-2.210427,-2.185823,1.607112,2.020737,7.056133,-8.540363,4.527292,-8.278841,4.519315,5.889779,6.444525,-8.869703,-6.020820,8.912911,-9.181459,-6.651880,-1.343948,-6.352771,-4.362281,-8.714847,-9.220650,-6.521101,4.469907,-0.756783,0.264078,-5.808555,-3.062977,-7.117270,-8.179903,-5.820985,-5.080967,-8.677660,-1.203488,-6.863150,-3.224806,-6.147497,6.862823,-6.395994,4.255383,-5.244253,7.627397,3.032468,6.158972,-8.583404,1.156992,-2.294294,1.693722,-4.529868,-8.606515,8.624312,-8.569237,3.737972,-8.264060,-4.550634,2.948833,-1.812011,1.880622,-9.505315,2.355733,-8.036563,-8.219292,1.390958,3.436126,1.565943,-4.460814,3.153635,-6.578808,3.061546,-0.794772,-7.894378,-8.015545,-0.010965,3.622922,-9.778696,-4.707329,9.798767,4.105814,-6.958298,6.156999,-0.404407,-9.267068,-7.959725,6.868696,4.309273,5.245646,9.503869,-8.744199,5.923283,-6.787050,0.534384,-9.608288,9.053044,6.436859,0.926670,6.708694,-7.384892,2.102414,-2.106454,-2.586065,-8.438395,-2.407472,7.854124,-5.909251,-1.022110,-0.479110,6.351577,-3.111797,1.830774,-2.825667,-1.667778,-4.227248,-2.291294,8.041955,-6.631183,-3.865251,9.369218,-3.729474,-8.826371,2.281388,-9.836903,-6.698542,2.988239,-3.405965,-7.905096,2.283631,7.896582,1.789986,7.746548,-6.270201,2.785202,-9.748584,8.488281,4.743555,1.460987,-8.641295,0.794846,7.317812,-6.136475,1.391222,7.589869,7.724370,-2.326485,-0.570917,-8.354157,-2.177147,5.555622,-8.033744,9.445947,-8.881474,2.387649,-9.183256,5.347098,1.989022,-2.900519,-6.696401,7.398616,-6.908022,2.761698,9.104701,7.362695,2.986973,-7.996416,-7.996750,0.457131,5.551702,7.564632,7.846227,8.581249,-0.962437,3.607227,3.457461,6.360958,5.123223,-8.169163,-2.053664,-8.763756,0.367738,0.022653,-0.549103,-1.011504,9.035159,-5.077803,8.114535,7.318023,9.794013,-4.167157,3.490885,-3.234302,-2.875288,9.489180,-1.352800,-7.619580,9.719899,9.304447,8.285786,8.968393,-2.377540,3.904001,2.629561,0.247243,-7.188072,-6.440085,7.344831,-6.539904,1.828470,-5.062233,9.367132,4.566435,9.248403,-5.510546,3.151589,2.713148,3.926905,6.673024,-3.216418,8.997247,-5.167338,-1.669209,-1.629457,-9.668570,9.215078,-4.230273,9.569093,5.992460,-3.202633,4.503090,7.014890,7.750452,-7.177385,8.898229,-2.534265,4.701616,-2.745870,-3.249950,-9.394504,3.868524,-9.064797,-5.426436,6.391846,-7.434510,5.978227,8.384083,-5.769262,-7.709854,1.553970,2.551437,-4.890313,9.245040,0.154273,-7.361237,-1.371541,4.333408,0.254620,-8.825362,1.256978,-2.275356,5.723012,5.321245,7.927132,-8.977930,2.605393,-8.846444,-9.270494,7.901013,5.358632,5.413731,7.992233,-9.147486,-7.698878,6.505152,1.993770,-8.639094,-7.749806,6.154407,8.247386,-9.649388,7.961266,2.152342,-1.889845,5.201065,-3.798957,2.241036,0.365071,-0.453982,6.737265,1.706389,7.064191,7.080780,3.591172,7.434056,-7.611121,-2.095079,-9.040403,4.837285,-4.719687,3.317784,7.253363,-8.209485,-8.209944,-8.570387,-1.065181,7.464144,0.040303,7.113734,6.006381,-3.541881,1.862780,9.418044,-0.598821,-7.027302,-5.136333,1.163635,9.157599,-3.112649,-2.652356,6.539127,2.070489,-6.166910,7.106625,3.560509,7.911888,-7.728750,-8.075427,-6.118478,-0.018601,1.440866,-9.250463,-8.521708,-8.550449,-6.321339,4.917052,-7.531545,-2.632030,7.931086,-8.815166,-3.635695,-9.503957,-0.712915,-5.121438,1.141849,-7.544820,7.838614,6.834639,-3.385000,1.451096,-6.433968,-6.206077,6.442133,-7.608220,-1.770361,6.651937,-2.241991,0.799983,3.640995,7.796308,-8.329141,4.119404,2.407093,4.199920,8.491136,-1.597349,3.051069,-5.862762,0.098222,8.645694,3.821773,-7.589360,7.583115,0.056072,2.012349,4.082689,-1.147728,-5.116689,-4.912083,7.708140,-4.742395,8.127533,-4.220794,-6.913424,-1.362615,2.633145,-2.660309,1.910602,6.319892,-6.442430,-5.945879,-7.154058,7.835899,-8.006615,-8.495381,5.482203,4.327129,-3.694704,-4.999482,9.599336,-4.610363,-3.490901,-6.906334,-7.221104,-2.721052,-5.640132,8.423806,8.121058,8.786061,-8.978542,3.075352,-2.730105,-9.284668,-2.632524,0.845849,-8.889402,-4.608465,5.776875,-3.285730,-9.038552,-9.012318,3.487723,1.291535,-7.070036,-9.747327,-8.902402,-8.753221,-3.973715,0.402031,0.533827,-6.274261,1.094071,3.127331,5.628170,-7.339732,-4.068224,8.198494,-0.137410,6.015638,4.566872,2.193749,1.823116,5.723976,0.723690],[9.817156,5.377099,-0.270183,-4.476007,4.250018,6.464306,-0.123968,8.146371,-5.807550,-3.368276,-9.481068,9.561450,-1.845585,-8.230679,5.804890,-8.858957,-5.625191,-7.913320,-6.854271,-4.605807,8.686990,-8.967053,-3.268241,9.565931,3.000996,-5.588318,3.560275,-8.441471,-6.267012,2.195910,-5.724352,-6.818027,-3.251147,9.251275,-0.799548,-0.568536,0.205894,8.508240,-2.318773,-6.846347,-5.014909,-8.010784,5.713499,9.964040,2.724915,2.355493,5.090740,-1.058291,-9.975004,-1.832266,-7.659662,-3.874420,5.879885,5.454782,-2.852381,-1.426518,-4.023513,0.826930,4.160877,8.771419,4.302089,-1.780361,1.299802,-0.132292,-5.146462,-3.527934,-1.374356,-2.939971,-4.563808,7.318964,-4.588170,0.549989,-8.176336,-9.038538,-8.731523,-2.493978,-0.554995,-1.442786,-3.576732,3.512862,5.716719,9.593984,5.229192,-6.622082,4.638872,-8.827596,2.679011,-7.162982,-4.026085,4.319931,-1.040933,9.364196,-6.649467,-9.254342,-4.838586,7.712299,7.563485,5.450493,-5.689164,-2.968317,-5.540721,3.693835,6.068551,0.445786,-2.395620,-0.188673,0.991488,0.106332,0.329339,-4.315355,-0.623193,6.729398,-1.690074,-8.005415,-6.254469,-5.872356,8.144177,-2.785864,1.143346,5.545384,-9.680101,-7.584190,1.524466,5.629175,-1.698848,-3.920684,2.102676,-2.222628,0.739993,3.553496,-7.669419,-7.404652,0.285150,-3.524718,-1.220389,7.053253,5.814337,-6.411876,9.391419,9.608534,1.918740,-0.834045,8.903311,-7.395273,3.316450,5.066615,4.237789,-7.566758,-2.275876,2.731868,-5.988311,-8.122484,5.970949,-5.372190,-0.990714,-6.023110,2.243647,8.967482,5.469065,1.522829,9.321572,-4.163977,-7.259578,8.067145,6.529663,-8.877412,7.528205,2.241293,9.117570,-4.790940,0.849343,-9.052237,1.242302,-0.744812,4.811414,-8.555243,-7.688153,-4.601938,2.768513,9.998487,-4.806821,-9.911620,-0.185572,-2.187224,0.167433,-9.826516,9.211160,7.647467,-6.445761,-2.854129,-5.298565,-2.196443,1.079594,1.231783,-9.326398,-7.619355,-9.276117,3.968596,-6.534684,4.852213,-2.979174,6.074580,-3.680035,-9.535178,-5.187252,-4.520540,9.067728,6.714514,2.307660,5.048126,3.696248,0.151131,-8.313550,4.121983,-3.174063,-5.825026,-8.792511,4.194373,1.972546,1.471230,-3.559520,-7.262996,-4.474624,-2.424972,-9.418255,1.560897,1.230224,-2.732275,4.603799,1.976650,-4.744400,-5.550504,2.996226,0.978215,-5.472162,-7.354334,-6.945559,-7.730329,-8.469074,7.825470,8.142932,-4.631621,4.042836,-1.449157,-6.817025,-0.859756,-0.515017,4.647368,-9.924028,-1.526552,8.167680,7.671171,7.635520,-5.078766,9.047651,-4.476939,8.079518,2.654672,5.547879,8.846735,9.044692,-6.845595,7.028910,8.931022,-6.578597,-6.834358,-6.095062,-8.317407,5.873511,6.610484,-8.531522,3.577690,7.388337,-0.476955,2.642784,-7.681336,5.580263,-8.604543,6.313648,8.773649,9.179840,8.050057,6.286871,5.033299,5.488307,0.104272,-9.480324,3.067724,-0.315926,9.068866,1.330977,-3.665385,-1.908282,2.207688,8.619743,-3.694953,-7.935831,-6.916886,3.574504,1.441836,1.318315,-0.269511,5.899290,0.850829,7.615116,3.203414,-2.739419,2.326881,-8.900470,-9.626259,3.987489,-7.683460,0.818938,-4.538860,-0.662090,4.270242,-7.359203,-0.106918,4.866489,8.744157,-1.084971,6.603608,8.348225,4.179842,-7.110486,7.704323,-1.820852,2.833174,-9.157561,-3.355649,4.006907,-3.729714,2.326325,-5.254701,-7.372891,2.175996,-9.575344,-1.964761,-0.873069,-8.940541,0.549041,-5.085092,2.076202,1.504781,5.697155,8.194489,3.356439,3.348193,-6.343452,1.276400,5.801998,4.563349,-4.954855,1.986632,6.220380,8.056577,8.233446,5.614157,3.213241,5.230781,-1.040051,-6.328285,-6.751948,9.164805,2.080660,-8.184583,7.903749,-3.300774,3.082221,-4.893146,9.580219,0.322554,-6.476897,-1.883051,-7.285843,-2.085319,-0.428053,-8.211037,-5.895880,0.640928,0.555989,0.048496,9.389425,-6.777639,6.907305,0.395929,-0.085677,-7.956445,9.532412,-3.582183,-5.503824,8.230566,5.005638,7.922080,2.309985,6.901863,-9.423273,2.362034,6.824856,-5.633109,2.881324,3.446423,3.678716,-2.599346,-1.560149,-2.474041,3.438079,4.055450,4.165532,4.993044,0.839689,8.835453,5.398219,6.409212,-0.543745,9.886007,-8.757554,5.999336,-5.372083,-1.272531,2.863219,4.622914,-1.496504,2.170564,-5.293528,-3.296941,-8.482047,3.403230,9.825490,-3.605802,8.080601,0.453987,-5.883609,9.082224,-7.895210,-3.123495,-4.505740,-5.952980,-4.705998,-7.785228,-5.799000,3.079613,9.323790,-0.471445,0.652920,-0.488102,9.859510,-5.685492,9.335234,-2.596803,-6.089854,3.090680,2.095005,0.450032,9.636982,-7.777769,0.946854,-6.972305,-8.759291,3.615091,2.122238,-7.082417,4.221266,2.835222,-6.273337,-1.147078,-6.064105,3.696073,9.480660,8.049934,6.435961,3.143173,8.843361,5.499548,-7.934128,-7.989487,-9.495552,-1.614583,5.784153,-0.214976,-9.618877,-6.443939,-3.973823,-9.583697,0.731531,2.371044,3.899595,9.097185,-2.201970,-4.923934,0.241683,0.445121,-5.477023,1.777026,0.888684,0.347663,-1.065469,-8.084592,-7.428134,-4.540645,-0.220372,-9.291112,-5.939759,5.845108,-3.398741,-7.525602,-4.166068,1.077851,2.258273,-9.303543,3.868114,-0.757998,-7.433166,-2.667718,6.475083,-1.387932,-2.399415,1.119622,-4.219201,-3.904678,4.007230,-2.742093,-5.853329,-6.667646,-0.568726,-3.543896,-4.190124,1.788860,0.915249,4.086921,-2.054974,-7.380231,-1.579567,-9.197123,-4.494172,4.347071,1.764095,8.339396,-1.749349,1.528354,1.560234,-8.755234,8.632977,-9.704248,8.481365,7.777234,8.971684,4.850323,-0.520930,-7.147915,-4.913168,-5.919415,1.249395,1.686734,4.707296,-6.947822,-9.894999,-0.521625,-0.470068,-9.655904,1.298900,2.845949,9.759643,-8.270352,-9.181609,2.333292,1.626711,-2.864591,6.327348,-4.096257,7.833685,8.573701,1.002159,5.898342,-1.435094,6.925584,9.385088,-8.754021,0.679477,-8.595508,2.861458,-5.273924,4.226775,9.630770,-0.645176,8.094844,-4.925009,1.990323,3.689937,8.922736,-2.523486,3.470450,-4.955565,4.755280,-4.078822,0.664507,0.126777,-8.522847,3.894614,3.464305,2.459711,6.512012,-6.863025,6.922751,7.632433,0.663637,-9.563733,8.933511,0.856421,-3.243766,7.565655,2.657335,-3.895516,3.114328,-4.820043,1.171778,-1.851510,5.572922,-8.933854,-3.643975,-5.440280,4.057011,-8.704281,2.303761,1.215829,6.159218,9.940556,5.044963,7.559951,3.219739,-0.915592,3.756518,3.611797,5.545442,-4.435600,8.217515,8.420459,0.718402,-9.158747,0.905515]], dtype = "float64")#candidate|11009|(2, 640)|const|float64
call_11008 = relay.TupleGetItem(func_10546_call(relay.reshape(const_11009.astype('float64'), [2, 640])), 1)
call_11010 = relay.TupleGetItem(func_10548_call(relay.reshape(const_11009.astype('float64'), [2, 640])), 1)
func_5416_call = mod.get_global_var('func_5416')
func_5418_call = mutated_mod.get_global_var('func_5418')
call_11025 = relay.TupleGetItem(func_5416_call(), 0)
call_11026 = relay.TupleGetItem(func_5418_call(), 0)
func_4455_call = mod.get_global_var('func_4455')
func_4458_call = mutated_mod.get_global_var('func_4458')
const_11050 = relay.const([6.493377,-7.045389,8.359823,-9.708885,-3.298451,8.126466,3.370640,9.196989,-5.752137,8.640833,-0.912760,-9.634241,2.306296,-2.217760,-6.769898,0.867763,3.103870,7.602714,3.407701,-7.398451,8.144992,3.647929,-3.466554,-1.286367,7.928757,4.542906,-6.212350,-6.352777,5.436559,-8.359448,-3.957821,8.489590,-7.950418,8.637206,-6.939996,6.958323,8.255866,-5.493317,-9.499961,-2.248224,-6.772622,7.041561,8.913988,-2.086364,-6.344334,-4.008837,-2.938443,-5.919703,-4.120705,-8.022721,2.509900,-6.033956,-2.870817,-6.315937,6.263317,-3.039245,-5.897695,4.771490,7.172161,-3.657177,-3.638396,1.458382,-9.640017,7.928364,5.621693,6.310228,-2.650694,5.899021,-4.868927,1.603116,-8.702547,4.429058,2.714090,-7.702855,6.446854,-6.959019,6.607275,-1.893932,8.769027,-5.705289,-5.487259,1.513732,1.612592,6.160598,0.391765,4.385077,1.211674,-1.998649,1.092668,7.805056,-5.296650,5.768357,3.196362,2.126932,6.927376,9.039825,-3.479454,-7.326442,2.278254,0.273540,-8.961317,-6.378840,-7.132271,-0.211344,2.333179,-5.750318,2.299605,8.612404,8.095591,3.900591,-4.675390,-6.894553,-2.268233,-4.440206,6.925004,-7.737607,6.294821,1.967778,2.836547,8.322423,8.715110,-6.335250,3.140614,-8.408254,1.211070,-7.569574,-6.983678,-7.601559,2.075541,9.950755,-9.555535,6.215776,-4.565226,7.158805,-1.443788,0.797363,-3.147645,1.178028,8.979645,1.828802,7.484590,0.297841,-8.462003,-7.515321,-7.318515,-9.977515,7.681938,-2.840770,-5.054304,-1.998913,0.376242,2.537104,9.299820,5.238548,-7.425468,4.369082,-5.647448,2.714229,-4.733439,1.887376,6.117703,-9.477813,8.124252,9.316063,-4.610447,-2.552414,4.862055,-4.964592,-2.943467,7.039042,-0.716911,6.832518,8.744493,3.026046,-1.728010,0.545217,7.611476,3.215486,-8.613768,-0.727090,-9.936517,-6.113844,-7.116650,-6.681764,5.019588,6.783818,2.541199,-5.956786,-4.470911,-9.113583,-7.066970,-5.281093,5.098045,-0.603464,-2.254527,0.730502,-7.868505,-1.476895,-1.439196,6.352743,0.421667,2.000365,9.603380,9.142608,1.387330,4.777863,6.538591,7.744679,4.566307,3.254307,4.122945,-3.175890,-9.231094,-6.993298,2.238866,-5.133004,5.462047,6.201439,-3.791862,-2.310604,9.359769,9.242845,4.877141,3.471117,6.282875,8.812038,5.573686,-0.259996,-0.383045,-7.614156,-9.227364,-6.153570,-5.642251,5.918109,0.901554,2.408829,6.775159,2.344574,9.859173,3.475717,0.471671,9.024236,-1.166240,5.729123,7.468757,-8.024461,-0.416401,1.411070,6.418824,8.273223,-6.594780,3.327337,-1.670002,-4.459247,4.546502,-4.329666,-8.384763,-1.430000,5.214424,-9.242864,1.179188,-2.790214,2.047714,9.700185,-4.551305,0.394073,1.650621,-1.102699,-3.686067,9.091706,-2.342043,-4.555201,-6.364052,6.190429,-3.705984,-1.541624,6.171676,-6.590733,-4.800607,6.410695,-2.230948,9.599705,7.604457,8.460381,-4.683731,2.805501,-2.116223,-6.491947,-0.059428,-3.447134,7.940384,1.957828,-3.993683,5.344606,3.684134,7.657671,-0.330943,0.124696,0.365816,7.834015,6.687633,-5.214901,-9.375354,-1.891154,6.310787,-1.221655,9.827993,4.020771,0.495208,-4.126196,2.878582,-7.815471,-2.039385,0.203938,0.128356,-9.528739,0.200248,-6.866587,2.357842,5.045036,-5.062128,1.218583,2.103294,8.308014,3.598410,0.387877,2.137420,-9.327855,0.476680,7.149174,-9.662286,5.257956,4.106928,-7.938154,0.271974,9.761026,7.945215,2.656518,0.913327,2.006756,8.972123,6.333261,-6.870300,-7.032137,1.690506,-9.727686,3.466452,5.886799,-4.325659,6.639064,0.365338,8.081973,5.109764,2.800684,0.635151,-4.735089,9.890651,2.271882,6.155455,3.603621,-9.682999,2.108808,-4.242435,-2.313842,-0.571345,-0.745779,-9.886388,-2.049306,-2.067506,-1.098665,-6.163828,-9.085947,-1.323952,9.999440,-1.982847,-1.449947,9.350397,-2.960169,-9.448546,-4.170293,8.243338,7.818933,-4.709793,3.724136,5.770945,-8.638629,-8.270388,-5.823333,-0.741657,0.478423,-3.228191,4.290816,8.881017,2.711224,-4.547491,-1.588383,-4.546873,5.768895,2.391405,-9.211114,-4.391008,-0.227268,2.598595,1.916949,-7.448573,-4.904613,-7.689705,-3.465916,-4.486680,-9.710710,3.816408,-8.314780,-3.278851,-4.138619,-4.432899,-9.688413,5.560586,-4.260160,7.628399,8.918213,-5.077285,0.665363,1.680513,-7.238405,1.335684,5.677542,4.306950,1.774569,1.985604,7.135462,-9.775964,7.481266,-0.330342,-6.406900,0.424211,3.453846,-2.182399,7.259621,6.335726,7.630089,-0.034491], dtype = "float64")#candidate|11050|(441,)|const|float64
const_11051 = relay.const([[3,4,4,5,-5,2,-2,-1,-4,8,2,2,-5,-3,-4,-10,6,-5,7,6,-4,4,-7,1,-6,3,-8,-3,7,9,-4,10,-3,2,-5,-4,5,7,4,5,-6,-9,1,-9,-7,-10,9,1,-5,4,1,10,-7,-10,-8,-1,3,-1,2,-3,-10,8,-4,-3,-3,-10,-6,-10,9,4,-7,2,-4,4,-6,-5,-3,8,-3,6,-8,10,-7,7,-2,-5,-7,-9,-3,9,2,-6,6,-2,-2,6,-7,4,-1,-1,-3,4,-5,-2,-7,-8,4,6,7,-10,-8,-1,-5,10,5,8,10,-7,-4,3,-10,2,4,-4,6,2,-2,-1,-7,2,-4,1,4,-9,6,10,3,-10,10,10,9,1,2,-8,-3,-10,-3,-1,8,5,-3,2,7,-5,-7,2,-4,7,5,6,8,-6,-8,7,-8,1,-9,9,6,-4,10,7,1,1,-9,-5,-2,9,-10,5,-1,1,-1,10,8,-10,-2,-2,3,10,-10,-1,7,-4,8,-5,10,-1,-10,-8,-4,5,-7,10,10,-4,-6,5,-4,-10,-10,1,5,-1,-2,-1,9,-4,-3,-6,6,-5,4,-8,1,-1,7,3,6,6,2,6,-3,10,7,-1,6,7,-6,-8,-7,9,3],[-3,-3,6,-6,1,-5,9,6,4,8,2,10,-2,-1,8,-3,2,2,5,2,-1,-4,-6,-2,2,6,5,-4,-5,-8,-10,6,8,1,7,-7,8,-9,-4,9,-1,10,4,-3,-2,2,1,-3,5,10,5,10,-1,-2,-10,-2,9,-6,-9,-10,7,10,-9,7,-7,5,6,-7,-4,3,1,4,-7,-3,-7,-7,4,-4,-10,-6,-10,9,6,-5,5,4,-5,-8,-9,9,3,-2,-3,3,-9,-10,-9,7,5,8,1,-7,-2,-9,5,9,-3,-6,-7,-10,7,-7,7,4,7,-1,5,5,3,8,-3,6,8,-2,1,3,-2,-6,4,10,-3,-10,-10,-4,10,-1,-1,-10,-8,-5,-4,1,-9,-4,9,4,8,-10,10,1,8,-6,-8,-7,9,-9,2,-1,-7,6,3,-7,8,9,-1,-2,2,-9,4,-7,10,8,-6,-3,-7,8,-2,3,2,8,10,-4,7,1,9,-4,5,-1,4,5,-4,3,8,6,6,5,5,8,-1,3,10,1,1,8,5,-9,6,6,10,4,-9,3,8,-8,-4,-1,10,2,1,9,3,6,1,-3,-8,-6,6,3,-3,-5,3,1,-5,7,5,3,-5,9,-4,9,-9,-9,6],[-1,6,-8,-1,2,-4,-3,4,-8,6,7,10,-2,-2,-8,-6,9,-6,-10,-8,4,-1,4,4,6,-9,-1,-10,-2,5,-3,10,2,-1,-4,5,4,4,1,9,-7,2,-9,-3,5,2,1,-1,9,-5,4,8,6,-3,-9,6,-3,8,-3,5,-8,-7,9,4,-2,9,-1,-4,-9,5,9,8,2,-4,-4,2,-5,3,-6,10,-5,6,7,4,6,6,3,-7,-9,-9,-1,-8,-9,-10,9,-5,-10,2,-8,3,-4,-9,-4,-1,1,-9,10,-9,-8,5,3,-2,8,10,10,10,-4,-2,-7,-5,-2,1,2,2,2,-9,-5,8,7,-1,-1,1,-4,-8,-8,9,7,-4,1,1,1,-6,1,2,-2,-2,-6,10,8,10,1,5,-9,-9,3,4,9,2,4,5,7,-8,-4,4,6,-7,-3,-7,1,5,-5,10,8,4,-4,-7,8,-5,-4,4,2,9,4,-10,-3,9,-1,-3,-10,1,6,6,-1,-8,-9,7,4,9,-5,2,-1,9,1,10,-9,-5,9,2,-4,7,3,-7,1,4,4,-5,1,8,5,3,-9,10,-2,7,3,9,7,2,3,-3,5,1,-1,-6,-7,10,-7,-3,5,5,4,-3,-2]], dtype = "int8")#candidate|11051|(3, 243)|const|int8
call_11049 = relay.TupleGetItem(func_4455_call(relay.reshape(const_11050.astype('float64'), [7, 7, 9]), relay.reshape(const_11051.astype('int8'), [729,]), ), 1)
call_11052 = relay.TupleGetItem(func_4458_call(relay.reshape(const_11050.astype('float64'), [7, 7, 9]), relay.reshape(const_11051.astype('int8'), [729,]), ), 1)
output = relay.Tuple([call_10999,call_11008,const_11009,call_11025,call_11049,const_11050,const_11051,])
output2 = relay.Tuple([call_11000,call_11010,const_11009,call_11026,call_11052,const_11050,const_11051,])
func_11055 = relay.Function([], output)
mod['func_11055'] = func_11055
mod = relay.transform.InferType()(mod)
mutated_mod['func_11055'] = func_11055
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11055_call = mutated_mod.get_global_var('func_11055')
call_11056 = func_11055_call()
output = call_11056
func_11057 = relay.Function([], output)
mutated_mod['func_11057'] = func_11057
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7656_call = mod.get_global_var('func_7656')
func_7657_call = mutated_mod.get_global_var('func_7657')
call_11130 = relay.TupleGetItem(func_7656_call(), 0)
call_11131 = relay.TupleGetItem(func_7657_call(), 0)
output = relay.Tuple([call_11130,])
output2 = relay.Tuple([call_11131,])
func_11140 = relay.Function([], output)
mod['func_11140'] = func_11140
mod = relay.transform.InferType()(mod)
mutated_mod['func_11140'] = func_11140
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11140_call = mutated_mod.get_global_var('func_11140')
call_11141 = func_11140_call()
output = call_11141
func_11142 = relay.Function([], output)
mutated_mod['func_11142'] = func_11142
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7894_call = mod.get_global_var('func_7894')
func_7895_call = mutated_mod.get_global_var('func_7895')
call_11143 = func_7894_call()
call_11144 = func_7894_call()
output = call_11143
output2 = call_11144
func_11199 = relay.Function([], output)
mod['func_11199'] = func_11199
mod = relay.transform.InferType()(mod)
output = func_11199()
func_11200 = relay.Function([], output)
mutated_mod['func_11200'] = func_11200
mutated_mod = relay.transform.InferType()(mutated_mod)
var_11206 = relay.var("var_11206", dtype = "float32", shape = (13, 7, 15))#candidate|11206|(13, 7, 15)|var|float32
uop_11207 = relay.tan(var_11206.astype('float32')) # shape=(13, 7, 15)
func_7547_call = mod.get_global_var('func_7547')
func_7549_call = mutated_mod.get_global_var('func_7549')
const_11225 = relay.const([7.949040,6.401594,1.151603,-7.548605,7.555520,4.596061,2.369961,4.925457,2.847428,-5.207700,-0.762228,4.753686,2.596626,-7.462902,-9.883307,9.661650,-5.296182,8.673879,5.200150,2.936060,6.095483,8.550083,6.457362,-2.776617,-2.452247,6.377927,-6.208232,5.683152,6.240462,-6.404903,4.775448,5.431277,-2.986066,-8.015041,0.741560,-7.126997,-7.679615,7.613377,-9.156650,-5.009067,2.157965,6.724798,-9.642675,9.390327,-5.571539,-3.989410,8.685241,5.397081,-3.344833,0.690742,8.318313,2.233050,-8.555182,9.829229,-8.615287,3.311525,-6.573064,5.470651,9.448291,-5.639747,-0.144004,-0.811121,0.628482,4.263120,6.671006,8.606933,-3.979013,-8.592679,5.674662,5.927771,7.505726,5.554315,7.760974,9.738412,-9.915294,-7.958466,-4.950174,-8.396888,-1.291342,4.030753,2.624280,-5.749622,6.989489,-7.230580,4.291375,-4.615203,-1.767316,-3.696926,-7.749505,8.622171,3.630356,2.076263,-9.549247,3.405634,2.665107,-1.002168,1.949531,-6.735773,6.107178,-0.730755,-7.433828,9.093572,-9.283496,-0.824618,4.337339,5.925917,0.161935,-6.290503,6.322693,-5.526604,3.411808,8.868717,-8.407570,4.490723,-1.757678,-8.831547,8.711087,2.740105,9.065867,-1.791661,0.786565,3.855562,5.000357,-6.943278,-0.189922,6.082040,0.533147,-8.375285,0.328373,8.061038,0.055622,-8.667079,-5.428668,2.254596,-4.298791,-1.243027,9.731454,-2.749250,-4.018415,4.393683,-3.882851,-1.685014,1.123886,0.033434,-0.021312,-5.934456,-8.270157,-5.242821,1.878755,-9.771758,-2.629263,0.123535,-9.038980,-2.764177,3.301954,6.420212,-8.894264,1.121336,1.952258,-0.103700,-0.625198,2.971435,-4.031487,2.039623,3.642058,0.130609,-8.227151,-0.362780,-1.399178,3.025750,-5.181016,0.670029,-8.589098,9.699791,3.837412,-1.373264,-6.686143,-3.610099,3.313586,-5.826129,4.609905,-7.640988,0.340271,8.504983,5.691304,9.664914,6.094380,7.292809,5.125353,2.132947,-7.358527,-2.710317,-4.007943,5.866372,-8.038011,-2.988032,4.264581,-2.187620,7.129585,2.083944,-1.748601,2.525830,4.191505,-5.326243,2.863150,-8.002008,9.597951,-4.967534,3.161983,0.527543,-4.046335,-2.046306,4.441904,-3.732524,-1.753849,1.369104,-3.380095,-2.644150,3.115224,-9.341703,0.208544,3.000561,-0.437221,-7.028391,6.430925,4.809885,7.368581,-8.229957,-5.975444,-6.627811,-0.876128,-9.363832,-1.004576,-5.232627,3.499383,-8.670958,7.277482,2.197217,1.763603,-4.471751,7.442213,-9.554367,8.186917,7.556267,2.266728,-2.379221,-8.207966,-2.850021,1.889253,-8.828877,-1.248113,1.534378,-2.308220,1.459652,-7.394286,2.507570,5.835991,6.561997,-6.330086,0.071364,-1.962668,6.724060,-7.538222,5.049317,-8.247884,-5.900402,-7.270473,7.108130,-7.088528,-9.108731,-1.661792,-2.929393,-2.072434,-9.006591,7.382604,-2.911707,6.915508,-4.534909,-0.043901,6.888933,4.082269,0.880597,-0.920312,5.756738,7.983510,7.645001,9.293244,1.023902,4.194383,-0.506324,7.881744,-3.035061,-5.028089,-9.361607,-0.574865,2.000931,5.789048,1.374011,3.814206,-9.039101,2.525222,-7.726970,-9.897338,-6.225486,5.944350,-4.836894,0.471705,7.719634,0.007717,-4.320276,-9.515617,8.671559,-1.643756,4.609001,4.477393,-9.507922,8.860004,8.832098,2.234565,-9.657533,-2.942817,1.002355,9.651970,2.793259,7.573730,3.960666,-6.746976,5.957360,3.382744,-0.278561,7.382094,3.730055,-3.630223,-3.352826,-7.468544,-4.391339,9.062891,3.194892,8.270150,3.609522,-7.260303,-9.145982,1.457439,-1.325926,-7.845813,-2.243235,9.199537,-4.258328,-7.870410,7.718132,4.119348,-6.195625,-0.515006,-4.304824,-5.923709,-3.963099,-3.829210,9.850669,-4.398133,-7.740874,6.630140,-7.805209,1.835191,-6.332473,2.859745,-5.619422,6.336417,8.084546,-4.497815,8.828257,9.762762,-6.583225,-7.424552,-6.571932,1.530503,3.671773,0.999977,1.126573,9.184770,-0.530920,-0.764622,0.222717,0.450453,0.405803,-3.647781,0.728931,-6.455110,1.747269,5.298333,3.618704,1.445015,4.127194,8.841350,1.461229,-6.859191,3.065342,-4.081451,7.693434,-0.011850,-7.272638,2.014162,-8.967466,1.836717,9.462982,9.840894,1.080456,6.908612,6.036660,6.036413,1.910700,6.483583,-8.997712,-0.425992,4.931524,-7.462893,8.693759,4.574689,1.131686,-1.076823,-3.657423,-7.047983,-7.854256,-0.115288,-2.893697,-0.547927,-9.708004,-0.465857,5.376774,-4.416939,3.237973,-6.342772,-1.315846,9.690859,-1.394453,-9.721486,7.065746,-3.525387,1.180313,-2.139100,0.199643,3.378686,-0.464394,-6.606232,-8.523485,-1.121838,-4.668448,7.962804,1.420546,-0.590470,-1.192411,-9.399799,8.544422,2.248349,0.171313,-3.074997,-2.379241,-0.250150,-9.702849,-4.310478,8.690565,-3.152119,-2.463686,7.045062,-2.054280,2.444439,-3.485209,0.257302,8.601241,-1.482146,-3.309931,0.460756,6.013228,1.768352,-7.076388,-2.551133,-2.754846,3.151606,-5.811967,-7.247720,9.577961,5.436719,6.708062,5.074884,7.843561,-1.154387,8.359871,-0.961193,7.025213,-0.373956,-3.364574,-8.297670,1.525653,4.883667,-3.929017,5.525694,0.605038,5.415608,4.347958,5.358816,3.709599,-5.180736,5.371109,9.087494,-9.470402,3.867948,-4.709340,3.557535,7.890545,-2.895188,1.738583,0.806542,-6.679344,5.396381,-8.315646,6.008009,-5.810833,-7.548898,-0.983485,4.970978,2.222795,2.388057,-2.530101,3.782742,8.808324,0.399113,1.462796,-2.990553,7.168059,0.766495,-9.287359,8.955215,-5.359390,7.333559,0.742689,8.496756,3.100062,4.706002,-4.636267,2.435081,-7.287531,-0.778733,-9.735597,7.761553,-1.838425,2.901957,4.763468], dtype = "float64")#candidate|11225|(546,)|const|float64
call_11224 = relay.TupleGetItem(func_7547_call(relay.reshape(const_11225.astype('float64'), [546,])), 2)
call_11226 = relay.TupleGetItem(func_7549_call(relay.reshape(const_11225.astype('float64'), [546,])), 2)
output = relay.Tuple([uop_11207,call_11224,const_11225,])
output2 = relay.Tuple([uop_11207,call_11226,const_11225,])
func_11230 = relay.Function([var_11206,], output)
mod['func_11230'] = func_11230
mod = relay.transform.InferType()(mod)
var_11231 = relay.var("var_11231", dtype = "float32", shape = (13, 7, 15))#candidate|11231|(13, 7, 15)|var|float32
output = func_11230(var_11231)
func_11232 = relay.Function([var_11231], output)
mutated_mod['func_11232'] = func_11232
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7737_call = mod.get_global_var('func_7737')
func_7738_call = mutated_mod.get_global_var('func_7738')
call_11242 = func_7737_call()
call_11243 = func_7737_call()
func_7480_call = mod.get_global_var('func_7480')
func_7481_call = mutated_mod.get_global_var('func_7481')
call_11250 = relay.TupleGetItem(func_7480_call(), 0)
call_11251 = relay.TupleGetItem(func_7481_call(), 0)
func_8623_call = mod.get_global_var('func_8623')
func_8625_call = mutated_mod.get_global_var('func_8625')
call_11255 = relay.TupleGetItem(func_8623_call(), 1)
call_11256 = relay.TupleGetItem(func_8625_call(), 1)
output = relay.Tuple([call_11242,call_11250,call_11255,])
output2 = relay.Tuple([call_11243,call_11251,call_11256,])
func_11259 = relay.Function([], output)
mod['func_11259'] = func_11259
mod = relay.transform.InferType()(mod)
mutated_mod['func_11259'] = func_11259
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11259_call = mutated_mod.get_global_var('func_11259')
call_11260 = func_11259_call()
output = call_11260
func_11261 = relay.Function([], output)
mutated_mod['func_11261'] = func_11261
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10804_call = mod.get_global_var('func_10804')
func_10806_call = mutated_mod.get_global_var('func_10806')
call_11386 = relay.TupleGetItem(func_10804_call(), 1)
call_11387 = relay.TupleGetItem(func_10806_call(), 1)
func_5063_call = mod.get_global_var('func_5063')
func_5065_call = mutated_mod.get_global_var('func_5065')
call_11407 = relay.TupleGetItem(func_5063_call(relay.reshape(call_11386.astype('float32'), [7, 6])), 2)
call_11408 = relay.TupleGetItem(func_5065_call(relay.reshape(call_11386.astype('float32'), [7, 6])), 2)
output = relay.Tuple([call_11386,call_11407,])
output2 = relay.Tuple([call_11387,call_11408,])
func_11412 = relay.Function([], output)
mod['func_11412'] = func_11412
mod = relay.transform.InferType()(mod)
output = func_11412()
func_11413 = relay.Function([], output)
mutated_mod['func_11413'] = func_11413
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4889_call = mod.get_global_var('func_4889')
func_4891_call = mutated_mod.get_global_var('func_4891')
call_11424 = func_4889_call()
call_11425 = func_4889_call()
const_11431 = relay.const([[[7.004322,-9.295475,2.603045,-1.445623,1.946659,1.170545,6.703062,0.071779,-7.746703,1.316813],[-5.849566,-3.407106,-9.107235,-6.026847,5.982912,4.925780,-3.013076,3.633492,-4.460999,-9.663845]],[[-4.300363,7.151085,8.532937,2.194577,-4.914880,-7.674376,-2.640103,7.162578,-3.649013,1.648144],[-7.697078,-3.415713,8.403672,-1.775459,8.166831,7.841777,0.329820,4.677015,6.376251,-2.600698]],[[6.790214,2.921199,-5.815600,-0.527166,-0.998966,-8.936979,-4.717335,3.066578,-2.305847,3.806375],[-7.949652,-8.150116,0.813385,1.274760,-3.875928,4.832605,-4.473405,2.475378,1.745024,5.697182]],[[-2.171199,-9.636939,-8.485497,-2.678314,-8.770282,-5.305742,-8.626941,-6.009560,-2.906338,5.073973],[6.550091,-7.118120,3.108098,-8.164589,3.641392,-5.997371,-8.158241,-5.673201,-7.816383,7.452061]],[[9.010003,-9.693472,-9.883679,7.502495,-3.610890,-2.905635,3.705920,6.989804,-0.667674,-3.389192],[8.419862,9.707577,1.188942,-3.322881,-7.359901,-6.961745,6.381361,-7.191794,0.056109,-4.524004]],[[-6.439027,-0.490554,1.751206,-5.123151,2.558607,4.308880,2.005082,-3.458166,-0.628806,-5.527518],[8.225623,-7.323079,7.760426,-5.041891,7.498562,-3.395593,3.808901,5.796511,-0.630202,6.529200]],[[-2.983757,-8.464963,9.049426,4.501694,6.244904,5.993550,-8.428736,2.326906,-8.855964,4.052020],[-6.551071,7.619375,-2.155402,-1.889463,-9.060734,-6.965735,-1.624701,-9.489932,-6.177481,-9.550676]],[[1.655266,-8.448575,3.248724,-6.351890,3.329842,9.130826,1.143617,-2.807431,-3.440960,3.234179],[-5.641524,4.433763,-6.137640,-7.530807,5.210649,2.674710,-4.369587,4.759208,1.583366,4.543779]],[[7.764971,0.073059,0.990122,-8.495813,-8.355516,-5.913437,1.077245,-1.352781,2.825759,-3.663005],[4.813604,-4.186960,5.095745,4.641324,8.637660,-0.637379,-4.138270,3.655429,8.134336,0.708953]],[[4.544216,-9.162132,0.656571,2.593917,0.138198,-5.625945,2.124436,4.047663,-7.523030,4.681998],[8.499785,-6.556436,3.350663,-1.878427,2.150270,2.544900,8.720925,9.764527,-9.952459,-6.105731]],[[7.987577,-0.673072,8.311593,1.263649,-5.310595,6.211759,-5.565612,5.590661,3.655459,-2.335476],[5.369860,-3.228042,3.030705,-9.307700,-9.853114,-5.078347,0.132249,-2.919070,-0.699884,-3.641218]],[[-6.136186,-3.266497,1.151660,3.720604,1.924854,6.628454,2.260256,-2.109566,0.991682,-2.775793],[6.232445,6.254018,9.258122,6.443825,-3.835215,-2.629584,9.727422,9.154837,9.580449,8.531134]],[[3.295233,6.437189,-5.359951,5.641524,-4.960886,0.337929,-5.892138,-7.131994,-3.813429,3.660738],[4.689144,-7.355322,6.986569,4.191810,-7.436520,8.291354,5.280372,-5.184789,-6.573561,-0.652802]],[[7.153998,-1.711623,-2.653945,6.104616,-6.830953,0.152838,-2.142198,-7.373376,-0.064565,-5.204299],[4.066920,-3.626775,4.590344,-5.945540,-3.175078,-3.797565,2.514488,-8.305717,-9.048781,-1.145374]]], dtype = "float32")#candidate|11431|(14, 2, 10)|const|float32
bop_11432 = relay.greater(call_11424.astype('bool'), relay.reshape(const_11431.astype('bool'), relay.shape_of(call_11424))) # shape=(14, 2, 10)
bop_11435 = relay.greater(call_11425.astype('bool'), relay.reshape(const_11431.astype('bool'), relay.shape_of(call_11425))) # shape=(14, 2, 10)
output = bop_11432
output2 = bop_11435
func_11447 = relay.Function([], output)
mod['func_11447'] = func_11447
mod = relay.transform.InferType()(mod)
mutated_mod['func_11447'] = func_11447
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11447_call = mutated_mod.get_global_var('func_11447')
call_11448 = func_11447_call()
output = call_11448
func_11449 = relay.Function([], output)
mutated_mod['func_11449'] = func_11449
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6146_call = mod.get_global_var('func_6146')
func_6148_call = mutated_mod.get_global_var('func_6148')
call_11477 = func_6146_call()
call_11478 = func_6146_call()
output = call_11477
output2 = call_11478
func_11491 = relay.Function([], output)
mod['func_11491'] = func_11491
mod = relay.transform.InferType()(mod)
mutated_mod['func_11491'] = func_11491
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11491_call = mutated_mod.get_global_var('func_11491')
call_11492 = func_11491_call()
output = call_11492
func_11493 = relay.Function([], output)
mutated_mod['func_11493'] = func_11493
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8235_call = mod.get_global_var('func_8235')
func_8237_call = mutated_mod.get_global_var('func_8237')
call_11522 = func_8235_call()
call_11523 = func_8235_call()
func_10200_call = mod.get_global_var('func_10200')
func_10201_call = mutated_mod.get_global_var('func_10201')
call_11532 = func_10200_call()
call_11533 = func_10200_call()
func_10179_call = mod.get_global_var('func_10179')
func_10180_call = mutated_mod.get_global_var('func_10180')
call_11550 = func_10179_call()
call_11551 = func_10179_call()
func_2016_call = mod.get_global_var('func_2016')
func_2019_call = mutated_mod.get_global_var('func_2019')
var_11556 = relay.var("var_11556", dtype = "float32", shape = (1980,))#candidate|11556|(1980,)|var|float32
call_11555 = relay.TupleGetItem(func_2016_call(relay.reshape(var_11556.astype('float32'), [11, 12, 15])), 1)
call_11557 = relay.TupleGetItem(func_2019_call(relay.reshape(var_11556.astype('float32'), [11, 12, 15])), 1)
output = relay.Tuple([call_11522,call_11532,call_11550,call_11555,var_11556,])
output2 = relay.Tuple([call_11523,call_11533,call_11551,call_11557,var_11556,])
func_11577 = relay.Function([var_11556,], output)
mod['func_11577'] = func_11577
mod = relay.transform.InferType()(mod)
var_11578 = relay.var("var_11578", dtype = "float32", shape = (1980,))#candidate|11578|(1980,)|var|float32
output = func_11577(var_11578)
func_11579 = relay.Function([var_11578], output)
mutated_mod['func_11579'] = func_11579
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6277_call = mod.get_global_var('func_6277')
func_6279_call = mutated_mod.get_global_var('func_6279')
call_11598 = relay.TupleGetItem(func_6277_call(), 0)
call_11599 = relay.TupleGetItem(func_6279_call(), 0)
func_8589_call = mod.get_global_var('func_8589')
func_8591_call = mutated_mod.get_global_var('func_8591')
var_11601 = relay.var("var_11601", dtype = "float32", shape = (330,))#candidate|11601|(330,)|var|float32
call_11600 = relay.TupleGetItem(func_8589_call(relay.reshape(var_11601.astype('float32'), [15, 11, 2])), 0)
call_11602 = relay.TupleGetItem(func_8591_call(relay.reshape(var_11601.astype('float32'), [15, 11, 2])), 0)
output = relay.Tuple([call_11598,call_11600,var_11601,])
output2 = relay.Tuple([call_11599,call_11602,var_11601,])
func_11615 = relay.Function([var_11601,], output)
mod['func_11615'] = func_11615
mod = relay.transform.InferType()(mod)
mutated_mod['func_11615'] = func_11615
mutated_mod = relay.transform.InferType()(mutated_mod)
var_11616 = relay.var("var_11616", dtype = "float32", shape = (330,))#candidate|11616|(330,)|var|float32
func_11615_call = mutated_mod.get_global_var('func_11615')
call_11617 = func_11615_call(var_11616)
output = call_11617
func_11618 = relay.Function([var_11616], output)
mutated_mod['func_11618'] = func_11618
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6366_call = mod.get_global_var('func_6366')
func_6368_call = mutated_mod.get_global_var('func_6368')
call_11628 = relay.TupleGetItem(func_6366_call(), 0)
call_11629 = relay.TupleGetItem(func_6368_call(), 0)
output = call_11628
output2 = call_11629
func_11633 = relay.Function([], output)
mod['func_11633'] = func_11633
mod = relay.transform.InferType()(mod)
mutated_mod['func_11633'] = func_11633
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11633_call = mutated_mod.get_global_var('func_11633')
call_11634 = func_11633_call()
output = call_11634
func_11635 = relay.Function([], output)
mutated_mod['func_11635'] = func_11635
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8623_call = mod.get_global_var('func_8623')
func_8625_call = mutated_mod.get_global_var('func_8625')
call_11677 = relay.TupleGetItem(func_8623_call(), 0)
call_11678 = relay.TupleGetItem(func_8625_call(), 0)
output = call_11677
output2 = call_11678
func_11688 = relay.Function([], output)
mod['func_11688'] = func_11688
mod = relay.transform.InferType()(mod)
output = func_11688()
func_11689 = relay.Function([], output)
mutated_mod['func_11689'] = func_11689
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9659_call = mod.get_global_var('func_9659')
func_9661_call = mutated_mod.get_global_var('func_9661')
call_11705 = relay.TupleGetItem(func_9659_call(), 0)
call_11706 = relay.TupleGetItem(func_9661_call(), 0)
func_8235_call = mod.get_global_var('func_8235')
func_8237_call = mutated_mod.get_global_var('func_8237')
call_11719 = func_8235_call()
call_11720 = func_8235_call()
func_8511_call = mod.get_global_var('func_8511')
func_8514_call = mutated_mod.get_global_var('func_8514')
const_11727 = relay.const([True,False,True,False,False,False,True,False,False,True,True,False,True,False,True,True,False,False,False,False,False,True,False,False,False,False,False,False,True,True,False,False,True,True,False,True,True,True,False,False,True,True,False,False,False,True,True,False,True,True,True,True,False,True,False,True,True,False,True,False,True,True,False,True,False,False,False,False,False,False,False,True,False,False,True,False,True,False,False,False,False,True,False,False,True,False,True,False,False,False,True,False,True,True,True,False,False,True,False,False,False,False,True,False,False,False,False,True,False,False,True,False,False,True,True,True,True,False,True,False,True,False,False,False,False,True,False,False,False,False,False,False,False,True,False,False,True,False,True,True,True,True,False,True,False,True,False,False,True,True,False,False,False,False,True,False,True,True,False,True,False,False,False,False,True,False,True,False,True,False,True,False,True,True,True,False,False,True,False,False,False,False,False,False,False,True,False,True,True,False,False,True,False,False,True,True,True,False,True,False,True,True,False,False,False,True,False,True,False,False,True,False,False,True,True,False,True,True,True,False,False,False,True,False,False,False,False,True,True,True,True,False,False,True,False,True,False,True,False,False,True,True,True,False,False,False,False,True,True,False,True,True,False,False,False,False,True,False,False,False,True,True,False,False,True,False,True,True,True,True,False,True,False,True,True,False,True,True,False,False,True,True,False,True,False,False,False,True,False,False,True,True,True,True,True,False,True,False,False,True,True,False,False,True,True,True,False,False,False,False,False,False,True,False,True,True,False,True,False,False,True,True,False,True,False,True,True,False,False,True,True,True,True,True,True,False,False,True,False,False,True,True,False,False,True,False,False,False,True,False,False,True,False,False,False,False,False,True,False,False,True,True,True,False,False,False,True,False,True,True,True,False,False,True,True,False,True,True,False,True,True,False,False,True,True,False,False,True,True,False,True,False,False,False,True,False,True,False,True,False,True,False,True,False,True,False,False,True,True,False,False,False,False,False,False,True,False,False,False,True,False,True,False,False,False,True,True,True,True,False,True,False,False,False,False,True,False,False,False,True,False,False,False,True,False,False,False,False,False,True,False,False,False,False,True,False,True,False,True,False,False,True,False,False,True,True,True,False,True,False,True,False,False,True,True,True,True,True,True,True,False,True,False,True,True,True,True,False,True,True,False,True,False,True,True,True,False,False,True,True,False,True,True,True,False,False,False,False,False,True,True,False,True,True,True,False,False,True,True,True,False,True,False,False,False,True,True,False,False,True,False,False,False,True,True,False,False,True,False,False,True,False,False,False,True,False,False,False,True,True,True,True,True,False,True,True,True,False,False,True,True,False,False,True,True,True,False,True,True,False,True,False,False,False,False,False,False,False,False,True,False,True,False,True,True,True,False,True,True,False,False,True,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,False,False,False,True,True,True,False,True,False,True,False,False,False,False,False,False,True,True,True,False,False,True,True,True,False,False,True,False,False,False,True,False,True,True,True,False,True,False,True,True,True,True,True,True,True,False,False,True,True,True,True,False,False,True,True,True,True,False,True,False,True,False,False,False,True,False,False,False,False,True,True,True,True,True,True,False,True,True,True,False,False,True,False,True,False,False,False,True,False,True,True,True,True,False,True,False,False,True,True,False,True,False,True,False,True,False,False,False,True,False,True,False,True,True,True,True,True,True,False,False,False,False,True,True,False,True,True,False,False,False,False,True,True,True,True,False,False,False,True,True,False,False,True,True,True,False,False,True,True,False,True,True,False,False,False,False,True,False,False,False,False,False,True,True,False,True,False,False,True,True,True,True,True,True,False,False,True,False,True,True,False,False,False,True,True,True,True,False,True,False,False,True,False,False,True,False,True,True,False,False,False,False,True,True,True,True,True,False,True,True,True,True,True,False,False,False,True,False,True,False,False,True,True,False,True,False,True,False,True,True,False,True,True,True,False,False,False,False,True,True,True,False,True,False,False,True,True,True,True,False,True,True,True,True,False,True,True,True,True,True,False,False,False,False,True,True,False,False,True,True,True,False,False,False,False,False,False,True,False,True,True,True,True,True,False,True,False,False,False,True,True,False,True,True,False,True,False,True,True,True,False,False,True,False,True,False,True,True,True,True,False,True,False,True,False,True,False,False,True,False,True,False,True,True,True,True,False,False,False,True,True,True], dtype = "bool")#candidate|11727|(945,)|const|bool
call_11726 = relay.TupleGetItem(func_8511_call(relay.reshape(const_11727.astype('bool'), [945,])), 3)
call_11728 = relay.TupleGetItem(func_8514_call(relay.reshape(const_11727.astype('bool'), [945,])), 3)
output = relay.Tuple([call_11705,call_11719,call_11726,const_11727,])
output2 = relay.Tuple([call_11706,call_11720,call_11728,const_11727,])
func_11742 = relay.Function([], output)
mod['func_11742'] = func_11742
mod = relay.transform.InferType()(mod)
mutated_mod['func_11742'] = func_11742
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11742_call = mutated_mod.get_global_var('func_11742')
call_11743 = func_11742_call()
output = call_11743
func_11744 = relay.Function([], output)
mutated_mod['func_11744'] = func_11744
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5178_call = mod.get_global_var('func_5178')
func_5179_call = mutated_mod.get_global_var('func_5179')
call_11767 = relay.TupleGetItem(func_5178_call(), 0)
call_11768 = relay.TupleGetItem(func_5179_call(), 0)
func_9202_call = mod.get_global_var('func_9202')
func_9204_call = mutated_mod.get_global_var('func_9204')
call_11789 = func_9202_call()
call_11790 = func_9202_call()
func_9127_call = mod.get_global_var('func_9127')
func_9128_call = mutated_mod.get_global_var('func_9128')
call_11793 = func_9127_call()
call_11794 = func_9127_call()
func_8511_call = mod.get_global_var('func_8511')
func_8514_call = mutated_mod.get_global_var('func_8514')
var_11797 = relay.var("var_11797", dtype = "bool", shape = (945,))#candidate|11797|(945,)|var|bool
call_11796 = relay.TupleGetItem(func_8511_call(relay.reshape(var_11797.astype('bool'), [945,])), 4)
call_11798 = relay.TupleGetItem(func_8514_call(relay.reshape(var_11797.astype('bool'), [945,])), 4)
output = relay.Tuple([call_11767,call_11789,call_11793,call_11796,var_11797,])
output2 = relay.Tuple([call_11768,call_11790,call_11794,call_11798,var_11797,])
func_11803 = relay.Function([var_11797,], output)
mod['func_11803'] = func_11803
mod = relay.transform.InferType()(mod)
mutated_mod['func_11803'] = func_11803
mutated_mod = relay.transform.InferType()(mutated_mod)
var_11804 = relay.var("var_11804", dtype = "bool", shape = (945,))#candidate|11804|(945,)|var|bool
func_11803_call = mutated_mod.get_global_var('func_11803')
call_11805 = func_11803_call(var_11804)
output = call_11805
func_11806 = relay.Function([var_11804], output)
mutated_mod['func_11806'] = func_11806
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10179_call = mod.get_global_var('func_10179')
func_10180_call = mutated_mod.get_global_var('func_10180')
call_11871 = func_10179_call()
call_11872 = func_10179_call()
output = call_11871
output2 = call_11872
func_11879 = relay.Function([], output)
mod['func_11879'] = func_11879
mod = relay.transform.InferType()(mod)
output = func_11879()
func_11880 = relay.Function([], output)
mutated_mod['func_11880'] = func_11880
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5143_call = mod.get_global_var('func_5143')
func_5144_call = mutated_mod.get_global_var('func_5144')
call_11921 = relay.TupleGetItem(func_5143_call(), 2)
call_11922 = relay.TupleGetItem(func_5144_call(), 2)
output = relay.Tuple([call_11921,])
output2 = relay.Tuple([call_11922,])
func_11958 = relay.Function([], output)
mod['func_11958'] = func_11958
mod = relay.transform.InferType()(mod)
mutated_mod['func_11958'] = func_11958
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11958_call = mutated_mod.get_global_var('func_11958')
call_11959 = func_11958_call()
output = call_11959
func_11960 = relay.Function([], output)
mutated_mod['func_11960'] = func_11960
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9072_call = mod.get_global_var('func_9072')
func_9073_call = mutated_mod.get_global_var('func_9073')
call_11964 = func_9072_call()
call_11965 = func_9072_call()
func_10179_call = mod.get_global_var('func_10179')
func_10180_call = mutated_mod.get_global_var('func_10180')
call_11966 = func_10179_call()
call_11967 = func_10179_call()
var_11984 = relay.var("var_11984", dtype = "float64", shape = (32, 40))#candidate|11984|(32, 40)|var|float64
bop_11985 = relay.left_shift(call_11964.astype('uint64'), relay.reshape(var_11984.astype('uint64'), relay.shape_of(call_11964))) # shape=(32, 40)
bop_11988 = relay.left_shift(call_11965.astype('uint64'), relay.reshape(var_11984.astype('uint64'), relay.shape_of(call_11965))) # shape=(32, 40)
output = relay.Tuple([call_11966,bop_11985,])
output2 = relay.Tuple([call_11967,bop_11988,])
func_11990 = relay.Function([var_11984,], output)
mod['func_11990'] = func_11990
mod = relay.transform.InferType()(mod)
mutated_mod['func_11990'] = func_11990
mutated_mod = relay.transform.InferType()(mutated_mod)
var_11991 = relay.var("var_11991", dtype = "float64", shape = (32, 40))#candidate|11991|(32, 40)|var|float64
func_11990_call = mutated_mod.get_global_var('func_11990')
call_11992 = func_11990_call(var_11991)
output = call_11992
func_11993 = relay.Function([var_11991], output)
mutated_mod['func_11993'] = func_11993
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10470_call = mod.get_global_var('func_10470')
func_10471_call = mutated_mod.get_global_var('func_10471')
call_12004 = relay.TupleGetItem(func_10470_call(), 0)
call_12005 = relay.TupleGetItem(func_10471_call(), 0)
func_3359_call = mod.get_global_var('func_3359')
func_3361_call = mutated_mod.get_global_var('func_3361')
call_12009 = relay.TupleGetItem(func_3359_call(relay.reshape(call_12004.astype('float64'), [2, 16, 2])), 0)
call_12010 = relay.TupleGetItem(func_3361_call(relay.reshape(call_12004.astype('float64'), [2, 16, 2])), 0)
output = relay.Tuple([call_12004,call_12009,])
output2 = relay.Tuple([call_12005,call_12010,])
func_12012 = relay.Function([], output)
mod['func_12012'] = func_12012
mod = relay.transform.InferType()(mod)
mutated_mod['func_12012'] = func_12012
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12012_call = mutated_mod.get_global_var('func_12012')
call_12013 = func_12012_call()
output = call_12013
func_12014 = relay.Function([], output)
mutated_mod['func_12014'] = func_12014
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5178_call = mod.get_global_var('func_5178')
func_5179_call = mutated_mod.get_global_var('func_5179')
call_12058 = relay.TupleGetItem(func_5178_call(), 0)
call_12059 = relay.TupleGetItem(func_5179_call(), 0)
output = relay.Tuple([call_12058,])
output2 = relay.Tuple([call_12059,])
func_12067 = relay.Function([], output)
mod['func_12067'] = func_12067
mod = relay.transform.InferType()(mod)
mutated_mod['func_12067'] = func_12067
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12067_call = mutated_mod.get_global_var('func_12067')
call_12068 = func_12067_call()
output = call_12068
func_12069 = relay.Function([], output)
mutated_mod['func_12069'] = func_12069
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8121_call = mod.get_global_var('func_8121')
func_8123_call = mutated_mod.get_global_var('func_8123')
call_12119 = func_8121_call()
call_12120 = func_8121_call()
func_5178_call = mod.get_global_var('func_5178')
func_5179_call = mutated_mod.get_global_var('func_5179')
call_12136 = relay.TupleGetItem(func_5178_call(), 0)
call_12137 = relay.TupleGetItem(func_5179_call(), 0)
output = relay.Tuple([call_12119,call_12136,])
output2 = relay.Tuple([call_12120,call_12137,])
func_12139 = relay.Function([], output)
mod['func_12139'] = func_12139
mod = relay.transform.InferType()(mod)
mutated_mod['func_12139'] = func_12139
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12139_call = mutated_mod.get_global_var('func_12139')
call_12140 = func_12139_call()
output = call_12140
func_12141 = relay.Function([], output)
mutated_mod['func_12141'] = func_12141
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9627_call = mod.get_global_var('func_9627')
func_9629_call = mutated_mod.get_global_var('func_9629')
call_12147 = func_9627_call()
call_12148 = func_9627_call()
func_8578_call = mod.get_global_var('func_8578')
func_8579_call = mutated_mod.get_global_var('func_8579')
call_12171 = relay.TupleGetItem(func_8578_call(), 0)
call_12172 = relay.TupleGetItem(func_8579_call(), 0)
func_9127_call = mod.get_global_var('func_9127')
func_9128_call = mutated_mod.get_global_var('func_9128')
call_12174 = func_9127_call()
call_12175 = func_9127_call()
func_5332_call = mod.get_global_var('func_5332')
func_5333_call = mutated_mod.get_global_var('func_5333')
call_12181 = relay.TupleGetItem(func_5332_call(), 3)
call_12182 = relay.TupleGetItem(func_5333_call(), 3)
func_12139_call = mod.get_global_var('func_12139')
func_12141_call = mutated_mod.get_global_var('func_12141')
call_12183 = relay.TupleGetItem(func_12139_call(), 1)
call_12184 = relay.TupleGetItem(func_12141_call(), 1)
output = relay.Tuple([call_12147,call_12171,call_12174,call_12181,call_12183,])
output2 = relay.Tuple([call_12148,call_12172,call_12175,call_12182,call_12184,])
func_12189 = relay.Function([], output)
mod['func_12189'] = func_12189
mod = relay.transform.InferType()(mod)
mutated_mod['func_12189'] = func_12189
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12189_call = mutated_mod.get_global_var('func_12189')
call_12190 = func_12189_call()
output = call_12190
func_12191 = relay.Function([], output)
mutated_mod['func_12191'] = func_12191
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10200_call = mod.get_global_var('func_10200')
func_10201_call = mutated_mod.get_global_var('func_10201')
call_12237 = func_10200_call()
call_12238 = func_10200_call()
output = call_12237
output2 = call_12238
func_12248 = relay.Function([], output)
mod['func_12248'] = func_12248
mod = relay.transform.InferType()(mod)
output = func_12248()
func_12249 = relay.Function([], output)
mutated_mod['func_12249'] = func_12249
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6875_call = mod.get_global_var('func_6875')
func_6876_call = mutated_mod.get_global_var('func_6876')
call_12343 = func_6875_call()
call_12344 = func_6875_call()
func_5807_call = mod.get_global_var('func_5807')
func_5812_call = mutated_mod.get_global_var('func_5812')
const_12379 = relay.const([4.337172,-3.171831,2.494895,-0.156587,-7.461058,-4.797260,-3.960214,-2.403232,9.563656,3.390353,7.753200,3.335671,3.292160,-8.873127,-9.750180,-2.455383,-9.855264,3.531375,-6.913480,-8.261878,1.620671,-0.948580,-9.143837,5.423040,-0.489571,-4.751041,-2.513855,9.542357,-8.988020,-1.230436,-7.064809,-6.239335,-6.567059,6.060027,0.354174,8.546233,-0.427246,-2.599985,-6.787410,9.774738,-8.854642,-1.106569,0.795699,5.118107,8.191613,8.732508,-2.778551,-4.356843,5.583021,-9.888862,-0.720934,4.117144,0.215195,6.090701,8.056386,9.534217,-6.216380,6.769573,6.483808,4.725669,-7.255223,5.844456,-7.839560,8.336305,6.292375,-3.171292,5.248644,-6.538422,4.320439,-3.178905,-4.575961,3.061601,6.883765,0.910416,0.220363,-8.205149,1.571959,-3.457645,3.185902,7.511814,9.173285,-8.604159,2.440517,-7.775093,6.823714,-1.062705,-7.052422,-1.696699,-7.206407,-0.084586,1.953698,7.654538,-3.132548,-4.427803,-9.002897,-2.697111,-1.622109,-3.161238,-1.201398,-0.345105,-3.910846,0.005603,-4.530696,-0.846816,0.053004,-9.081894,7.875095,5.879618,-3.354541,-3.315245,2.160714,-5.291375,2.695438,-3.751528,4.232579,2.625634,5.467934,-2.023855,3.376946,-4.896273,-4.403620,7.602785,-7.583389,0.801057,-1.213911,2.289608,9.453750,4.482563,1.651150,-0.948239,9.787784,-2.023455,-4.582926,-8.655806,7.411980,5.861372,-4.448461,-9.084052,7.805461,1.504030,7.927445,-6.733847,-4.435469,1.869116,3.366293,0.559513,-0.410784,9.318119,-3.553609,-3.840051,-5.623023,4.687637,-4.226766,3.924576,-1.080067,6.682428,-7.238604,-4.961531,-1.462174,-8.055744,5.500756,-2.553048,8.861259,-6.564775,7.476830,4.328512,-7.824884,-5.038628,-8.669826,8.898264,-3.129597,-4.588477,-4.752292,-0.832317,8.513833,-4.705000,-0.890995,-6.377484,-0.719342,4.838442,9.292907,1.289341,-1.381593,-8.643643,2.457906,4.886364,-1.808619,-5.688254,-4.937993,3.898056,5.367609,-8.632661,-2.622832,8.870901,3.845715,1.480337,-2.416965,5.608101,-4.135754,2.492870,8.819907,-7.731627,8.385132,-3.630969,0.890062,0.813733,3.285625,-6.946995,-1.038574,-9.773399,-1.600536,-1.486058,-3.143934,3.280454,1.145189,-8.451833,4.383408,-6.746998,-4.276330,-7.258373,9.223419,-0.085122,3.786894,-1.098000,2.130937,-5.223300,-8.969763,-0.815016,-6.001959,-6.177498,-0.121086,-2.721541,1.638354,-9.632943,1.349693,5.932251,-0.825223,7.570787,-3.649781,8.641362,-0.426173,7.348869,-3.570703,1.650929,-7.900406,-7.907305,-6.033209,-3.667279,-7.073800,5.745229,-7.897935,-9.342472,8.618768,-4.638333,6.574551,0.568045,3.413021,2.063455,-0.984660,-2.906537,-0.121976,1.638284,0.315864,-6.020610,7.085010,6.123009,-8.603624,-4.577772,9.509342,9.215277,3.276990,-7.111074,0.692943,-5.430873,-0.927876,-0.908856,-5.674828,-1.796465,-5.775460,1.461548,-2.525402,9.700386,-6.051140,-3.231413,-1.203793,-2.048903,-6.270378,6.434990,-7.731025,6.230019,-4.277370,3.501914,-5.715762,3.610025,-3.926534,-6.323108,0.835970,7.063471,9.311192,-7.388888,7.738577,0.290789,2.223113,-5.017268,9.000234,0.009794,1.834195,-1.899828,-1.514733,-7.901677,-8.276955,8.711197,-1.305881,5.400583,1.303191,-7.731083,-8.550704,-8.658830,-8.564018,-5.085577,5.324279,4.149368,-8.052358,-8.594024,9.829163,-1.821090,6.975283,-1.390489,-2.151823,2.725533,0.701013,0.091714,-8.607332,8.458299,-2.953620,-2.416089,6.577097,-2.339181,2.816825,-8.445752,-8.427243,-1.075736,2.935338,9.669034,-1.508034,1.895217,-6.236202,0.738842,-6.010340,0.206373,-2.567899,-4.981935,-4.115707,-1.783410,6.443549,-7.899695,0.973589,3.559161,8.624242,-2.959629,-4.053332,4.916802,-5.141547,0.382743,-8.592023,1.349539,-2.558063,-1.388758,3.994874,-0.355384,-7.846060,-9.207267,9.832448,-0.037332,-7.820601,-3.068256,-5.879320,-2.148820,-8.503269,3.835070,-9.612955,-5.333006,3.035183,8.982071,9.651083,3.755883,5.162633,8.076109,-4.031495,7.092943,-1.806022,4.479960,-4.199837,-9.807576,7.131600,3.858804,-4.730129,-4.875478,-1.297767,-7.212868,7.710169,0.233250,6.134310,5.231415,-5.822069,4.128176,-9.570799,2.218765,-4.237088,8.972200,-0.253175,8.708463,-2.147795,-7.244243,-2.401448,-3.689920,-8.766551,-3.207077,-8.808032,5.507216,-2.404479,-4.018505,8.944171,5.575613,0.420585,7.464103,7.102245,-2.940998,-0.748032,1.075611,-5.906335,-2.844398,3.958129,-2.638098,-8.869508,0.005476,-5.618876,2.360882,5.661099,5.589813,-5.634654], dtype = "float64")#candidate|12379|(441,)|const|float64
var_12380 = relay.var("var_12380", dtype = "int8", shape = (729,))#candidate|12380|(729,)|var|int8
const_12381 = relay.const([4.566942,0.064255,-3.839231,-6.895566,4.514581,4.576190,5.006107,-5.829443,-0.925682,9.875439,-6.232586,-8.753624,1.523713,-3.255875,-2.917462,-9.524337,8.702176,4.082904,9.837597,8.822876,4.851423,7.160764,6.393070,7.200584,1.993795,-7.122237,5.762799,-0.696050,7.119432,2.316666,-6.002447,9.329554,7.187388,7.337412,-5.742846,-2.977781,-6.594402,9.691390,4.213490,7.776472,5.795499,-3.676898,-3.584742,7.532194,6.712841,-3.857985,-2.939977,-4.506554,-7.753749,2.070717,-6.556232,-4.911358,6.897304,-7.925037,-5.669367,6.127047,-7.235413,-4.303622,8.544178,-8.995228,0.016348,-1.331395,-9.789466,3.832418,6.987006,5.632187,-8.490867,-9.171509,4.510236,-1.288372,-4.300661,-3.918159,-7.881782,0.338716,-7.823428,-9.961591,-7.617813,-6.550014,9.459465,5.104141,-7.464287,-3.221654,-5.093022,-8.434083,-9.283961,-4.875292,-7.367105,5.237260,9.464494,-3.199168,-2.581126,6.247935,-3.201721,2.325486,-5.226100,-4.996584,0.803214,5.490350,7.323839,-4.396939,2.982238,-9.344293,1.760717,-6.628698,-9.956435,-2.696134,0.413890,-2.917751,3.658287,-8.508732,-7.348321,-3.938251,2.984189,-2.857147,5.042790,-6.048224,5.299549,6.533064,-9.191230,7.540102,-6.145098,6.817553,-8.590912,3.294987,0.601111,0.287904,-2.933227,-2.641875,2.752197,-7.646541,5.190912,4.285813,9.514776,8.122685,-2.663265,2.789949,-5.157092,8.289139,-9.146864,5.001559,-1.369569,8.415169,-9.537038,-4.759938,-3.930384,4.716850,-1.980575,-0.727626,5.177123,9.053383,0.788469,-4.116275,-5.824927,-8.949939,6.597675,-8.387438,-4.912881,-7.730669,6.640629,-3.365472,-3.517104,0.755312,3.587614,-2.932422,6.518444,-7.316508,-3.505071,8.598979,-5.366329,8.412237,9.747732,5.721194,-3.415535,-4.839284,0.884276,5.826653,-4.705213,-1.518015,1.469108,-7.891961,-9.728045,2.835818,-8.803202,4.440443,-7.238895,-9.235012,2.894864,4.392316,3.944812,3.220582,-0.953602,-7.544620,-3.139780,-9.151030,-2.553333,-6.615428,-6.502432,1.752120,-9.767990,3.850691,-0.608578,-7.079167,-4.424760,9.795289,-1.685052,-1.415712,3.056768,3.574566,1.214243,6.546825,5.016439,-3.459969,3.823838,-0.817918,0.828934,-0.564754,7.671531,1.863881,-2.742867,6.078093,3.024216,-5.602750,6.975722,-0.026926,5.492655,-9.104116,-0.395913,-6.806022,-3.348524,4.869344,3.031820,2.369506,3.918104,-2.916375,8.953988,-1.103486,0.439853,-4.428431,-9.540659,9.422755,-5.974282,-6.519920,-8.693527,8.220357,4.298631,-7.677172,-6.900236,6.142461,-0.313613,-5.592198,-6.009273,2.630709,5.014193,-5.723389,-3.220774,5.976487,4.985716,8.543837,9.477480,-6.652645,-8.221025,6.288424,0.926387,-7.976591,-0.864455,3.447503,0.271711,-2.617468,-0.849752,-4.626117,-6.712489,-6.595925,-0.047517,1.449948,-5.082346,-6.259775,7.756930,9.731785,5.606483,-1.034150,-1.472286,-7.258498,-2.553079,-3.144633,9.772235,-9.846197,7.928761,2.698995,-0.772149,8.580403,-7.016189,9.429072,7.779355,-6.013777,-0.694621,-7.489943,3.668520,-9.670959,0.519629,-3.831540,-1.824492,-5.094046,2.683814,2.331517,-2.805829,-0.426330,-2.497077,-1.619878,5.957019,-1.042669,-9.988256,-7.997963,0.676405,5.991166,-8.563061,3.724986,-7.678428,-1.018170,7.100335,-8.003052,-1.799242,9.471949,-9.896848,-0.001856,-3.920646,-6.084044,0.398808,-9.712273,0.551599,2.569620,-9.198096,-2.317174,-6.989482,-9.532224,-8.384412,7.637282,-9.482826,-7.805249,5.550246,-6.564620,3.557107,-7.584676,4.070038,-7.666047,-2.322978,-7.587512,-2.441268,-4.876447,4.018069,1.171107,0.576353,-2.665891,1.187290,-2.756542,0.783707,-1.515344,2.202037,4.136194,-6.446184,-2.999473,6.740443,8.249973,7.889678,-0.410653,-2.087468,3.101836,4.368947,9.610934,-3.884304,9.785228,-9.757426,-6.061121,-2.490334,9.978277,4.664773,3.974819,-1.044297,7.443268,-3.413402,7.891322,9.663616,-5.126406,-0.063404,1.778038,-0.652200,6.284431,-0.845948,-5.073994,9.365904,4.764793,9.891702,-6.686346,1.808655,-1.990047,9.667433,2.718656,-3.827983,6.962664,8.271680,0.336979,3.406907,5.330077,-5.295389,-6.021344,1.392966,3.265964,6.144549,-9.883806,7.796062,-2.065173,-4.174524,6.200662,-5.459086,-4.466272,6.774593,6.202452,-4.026144,-1.545340,-7.452844,-9.421188,5.678011,2.589180,7.341382,1.029779,-1.577151,-2.967378,2.069898,8.956880,-4.222830,-6.090572,8.447052,6.798197,-2.478395,-4.946528,-8.454747,-6.336661,-1.581749,-0.660087,0.216038,-3.963626,-7.295058,1.566077,-4.170524,5.869952,-6.860858,-8.187620,1.533893,5.673258,-8.337675,-0.179951,-8.668189,-6.295209,-3.993397,5.760102,-2.077760,6.136749,-0.678328,4.050010,7.771777,9.595259,-5.932116,1.643167,1.275042,3.413911,8.875478,2.325429,-4.542796,-9.715858,-1.835493,-9.429057,-8.870726,0.912935,-5.693422,8.652105,3.788115,1.383698,3.107579,6.174677,6.679203,-2.122429,2.472658,4.721235,4.869163,-8.791660,1.585041,-1.276578,9.211606,-3.454879,-7.901853,9.493406,-8.566499,8.015312,-9.608073,5.244697,4.944694,-6.026495,5.971719,-5.816647,9.283831,-5.658192,-1.611106,-3.163183,6.002812,7.090189,2.810760,4.473609,-6.268070,0.171022,-3.001388,-0.744805,-5.213493,-4.456317,-7.296779,-1.555552,7.945582,-7.399877,0.487647,7.222149,2.930773,9.478111,-7.324721,1.900446,4.020557,7.582526,1.481515,-8.589253,-3.259883,7.201973,8.197130,-2.138761,-6.055718,-0.198020,-6.459619,-5.156732,5.479176,-2.905170,7.609920,-6.039166,-9.890926,-7.705074,4.814881,6.787028,8.358535,4.511061,-3.055239,0.223122,-8.192764,-9.165072,-3.382589,7.093924,-5.161721,0.776024,8.593018,6.859006,8.859781,0.831139,-6.178849,-7.221163,0.445119,-0.741395,2.076341,-4.911552,-9.390004,-3.940633,-6.843235,-9.216373,9.277801,-1.160888,1.088034,-8.634651,6.375787,2.649378,-3.892580,7.332083,3.734144,1.617345,3.627131,8.441592,4.859259,-6.208231,0.579031,-8.682112,-6.968078,4.995940,-2.737039,-1.558683,0.006735,0.383910,4.297735,9.890215,5.062816,-2.470779,-3.244945,-4.571790,3.337745,2.699480,5.178145,-5.909231,-6.704717,7.717992,2.208182,-8.742438,8.406634,4.605643,-8.692831,-7.320216,8.166536,0.382573,-6.389363,6.296156,-5.785991,-8.109221,9.904513,-8.207267,-0.214760,-6.039743,5.015812,-6.577682,0.935381,-7.092860,-9.133092,7.423942,0.883301,9.038543,-7.126214,3.010935,-4.084628,-2.076429,-6.534506,7.350265,-0.297302,3.502333,3.218129,-1.327353,0.962441,-3.188238,4.454213,2.876090,-2.382071,-0.890962,-2.226322,-1.060153,9.838876,-5.744088,8.196342,-6.365637,8.064609,0.737052,9.251458,6.626573,-3.567493,-2.988415,3.862580,-4.613251,6.080070,-7.539526,2.679073,-1.263163,4.677825,-0.167000,-9.699456,-5.742025,-6.128152,-3.768545,-1.036799,-1.661676,-1.662732,-2.560785,6.398204,8.885453,-8.005070,-1.322407,-6.043981,-5.670425,4.105642,-4.357596,-9.813421,4.436143,0.008582,-7.899394,-8.740766,-8.258894,-8.795017,-2.331869,-5.233926,-8.623406,-1.588521,-2.347332,3.621359,-8.711656,9.130493,-2.702248,-1.316478,7.560849,-9.647248,8.761079,-8.541533,9.195131,-4.146089,-7.937222,6.116237,9.218994,8.268197,7.439939,9.995506,7.882262,-0.448666,-7.720151,-8.996644,-8.447654,4.446569,2.325173,-6.562592,-3.745927,-9.409291,9.911027,5.902923,5.306411,-6.338787,0.328914,2.194655,-7.937826,8.623227,-6.157417,-4.449293,-6.788862,-7.160167,-4.357034,-8.787833,0.011694,2.144654,-5.499968,-0.907114,-1.680601,-0.608804,-6.912156,-7.455117,-5.867263,3.162353,8.369657,8.797249,0.486217,1.755678,-2.240807,6.973580,2.073010,-0.643750,0.388777,2.633346,9.843648,-5.767428,2.148063,0.214779,-4.432840,-0.833357,-9.043598,0.186202,5.243432,5.374209,-9.148587,-3.427423,-5.668531,8.004108,0.812838,1.837141,5.735286,-0.691184,-4.186378,-6.685244,-7.795020,0.420263,-9.562704,-3.280739,3.308497,-2.538302,-8.384584,4.850041,0.494776,2.542198,-2.141343,-4.071900,-3.600105,-3.010474,-0.938249,7.796985,-4.386569,0.466476,9.902997,-7.238039,-1.050985,-2.202916,1.216659], dtype = "float32")#candidate|12381|(792,)|const|float32
call_12378 = relay.TupleGetItem(func_5807_call(relay.reshape(const_12379.astype('float64'), [441,]), relay.reshape(var_12380.astype('int8'), [729,]), relay.reshape(var_12380.astype('int8'), [729,]), relay.reshape(const_12381.astype('float32'), [792,]), ), 8)
call_12382 = relay.TupleGetItem(func_5812_call(relay.reshape(const_12379.astype('float64'), [441,]), relay.reshape(var_12380.astype('int8'), [729,]), relay.reshape(var_12380.astype('int8'), [729,]), relay.reshape(const_12381.astype('float32'), [792,]), ), 8)
func_6251_call = mod.get_global_var('func_6251')
func_6253_call = mutated_mod.get_global_var('func_6253')
call_12383 = func_6251_call()
call_12384 = func_6251_call()
var_12387 = relay.var("var_12387", dtype = "float32", shape = (1980,))#candidate|12387|(1980,)|var|float32
bop_12388 = relay.floor_divide(call_12378.astype('float32'), relay.reshape(var_12387.astype('float32'), relay.shape_of(call_12378))) # shape=(1980,)
bop_12391 = relay.floor_divide(call_12382.astype('float32'), relay.reshape(var_12387.astype('float32'), relay.shape_of(call_12382))) # shape=(1980,)
output = relay.Tuple([call_12343,const_12379,var_12380,const_12381,call_12383,bop_12388,])
output2 = relay.Tuple([call_12344,const_12379,var_12380,const_12381,call_12384,bop_12391,])
func_12399 = relay.Function([var_12380,var_12387,], output)
mod['func_12399'] = func_12399
mod = relay.transform.InferType()(mod)
var_12400 = relay.var("var_12400", dtype = "int8", shape = (729,))#candidate|12400|(729,)|var|int8
var_12401 = relay.var("var_12401", dtype = "float32", shape = (1980,))#candidate|12401|(1980,)|var|float32
output = func_12399(var_12400,var_12401,)
func_12402 = relay.Function([var_12400,var_12401,], output)
mutated_mod['func_12402'] = func_12402
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8918_call = mod.get_global_var('func_8918')
func_8920_call = mutated_mod.get_global_var('func_8920')
call_12414 = func_8918_call()
call_12415 = func_8918_call()
output = relay.Tuple([call_12414,])
output2 = relay.Tuple([call_12415,])
func_12454 = relay.Function([], output)
mod['func_12454'] = func_12454
mod = relay.transform.InferType()(mod)
output = func_12454()
func_12455 = relay.Function([], output)
mutated_mod['func_12455'] = func_12455
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8623_call = mod.get_global_var('func_8623')
func_8625_call = mutated_mod.get_global_var('func_8625')
call_12492 = relay.TupleGetItem(func_8623_call(), 1)
call_12493 = relay.TupleGetItem(func_8625_call(), 1)
output = call_12492
output2 = call_12493
func_12507 = relay.Function([], output)
mod['func_12507'] = func_12507
mod = relay.transform.InferType()(mod)
output = func_12507()
func_12508 = relay.Function([], output)
mutated_mod['func_12508'] = func_12508
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6088_call = mod.get_global_var('func_6088')
func_6090_call = mutated_mod.get_global_var('func_6090')
call_12562 = func_6088_call()
call_12563 = func_6088_call()
func_11447_call = mod.get_global_var('func_11447')
func_11449_call = mutated_mod.get_global_var('func_11449')
call_12581 = func_11447_call()
call_12582 = func_11447_call()
func_9004_call = mod.get_global_var('func_9004')
func_9008_call = mutated_mod.get_global_var('func_9008')
const_12585 = relay.const([8.989911,-4.068109,-8.502126,8.658544,-7.695983,-1.865926,3.225181,3.711404,5.151607,-1.888040,5.112900,6.964621,-0.860909,6.159855,-1.356039,-6.790860,5.991794,-2.349089,-4.321072,-0.788872,2.825296,-0.630988,-6.732369,1.316638,9.249607,3.671021,-2.883134,-5.322601,-6.706578,-8.006064,2.903023,6.911430,-4.681049,-2.413251,2.213615,2.636703,-0.726178,2.217182,7.521501,-6.332074,-4.026227,-3.403334,7.545143,-2.186471,-6.554944,-5.426170,7.628910,-2.232694,-9.826750,-7.201639,0.436848,-3.881476,9.792641,-7.717063,-7.182320,7.525212,-3.778017,8.370114,-9.457304,-4.369668,7.263385,3.677188,-6.248673,-0.534014,-3.267972,-0.932695,-9.107598,6.324491,-2.977675,-9.362215,2.932179,7.098553,3.219601,8.659753,-3.221188,-4.811411,-6.040621,-9.134717,7.590992,8.961355,2.641390,4.940262,-2.750577,1.711546,8.559254,-5.889011,-4.587072,-0.451292,-4.542260,6.723020,2.493304,-2.584185,6.393426,-9.284779,1.460794,7.699330,0.915481,-6.640640,6.433151,-0.868608,-5.392667,7.473569,1.552409,-1.279068,1.037898,-3.554874,-2.715901,5.379709,-0.982559,-7.085533,3.980666,-7.095774,6.313013,1.208893,2.490902,4.867390,-0.388240,-1.378158,-4.314202,2.735310,-5.592071,1.619014,0.737871,5.762773,-3.538721,9.243579,7.191779,2.646707,5.661335,1.527114,-0.468512,-9.991247,-4.833746,-6.819061,2.717580,-3.021439,-7.728971,-1.590218,-5.227897,9.332532,-7.336008,7.948562,-7.427914,-2.029154,6.450597,-8.226443,-9.732517,-2.502292,0.638985,3.069749,-5.537408,6.999406,0.160648,-2.218049,0.611143,7.596384,1.641645,1.069269,7.655632,3.588404,3.778463,-0.506370,-7.411130,-1.421619,-9.633491,-2.551689,-1.962628,2.853124,-3.989230,1.973313,6.697416,3.309214,0.689552,-9.978742,-5.036978,8.403698,-2.763282,6.499919,-3.768615,-1.286532,1.076258,-4.095349,-5.641886,-4.782818,2.663834,-1.363951,-1.822886,7.559218,-5.592895,-1.058630,6.811678,1.137388,2.973293,-3.028710,-6.098708,-4.004294,-1.524311,0.195388,-4.039951,8.950882,-7.487115,-7.856626,8.159736,-6.895240,-6.544430,-6.526159,8.694176,0.543049,-8.476556,-8.852062,-6.013807,-5.373958,9.491100,-7.163377,4.475159,-3.539116,8.845700,-0.516806,-8.269987,5.027359,0.441758,0.459504,8.101220,-5.735363,-2.473042,-5.979224,2.041495,0.898659,7.243083,7.409439,1.810166,-7.671174,-7.636679,1.081620,2.215837,3.470188,-2.808281,6.592551,0.621035,8.675714,0.472060,7.271798,0.942000,0.645251,-7.558236,5.498358,-4.967474,-8.142595,6.152491,-0.044232,-1.347851,5.596483,-3.448158,-3.679050,-3.389595,5.164806,-6.733924,2.633831,-1.213995,-3.630809,5.174685,-9.870761,0.718299,-6.642150,7.332707,0.677253,2.017802,-8.286920,-4.996865,6.683674,9.760297,-0.470302,0.362343,0.295067,-7.210793,-6.310925,3.373892,-6.908084,-6.800766,8.592563,2.923323,4.838160,-8.862303,-0.850187,5.812422,1.164807,-0.452504,-7.327415,5.452613,3.941133,2.996223,1.406915,1.643251,3.377017,5.138856,7.408086,7.644560,0.282935,-9.046127,-1.349745,2.903877,-2.955944,8.699386,-7.844072,-1.667050,0.329922,6.524398,7.603679,-6.900457,-5.569147,9.985650,-0.366181,2.300307,-3.668384,9.752778,3.402399,2.040185,-8.958861,-4.417025,6.495038,6.526055,-5.677776,-4.086375,0.771594,-7.831367,-2.598846,1.387164,-1.372912,-2.409355,-2.878631,-2.401631,-4.154468,-2.889587,-9.534035,1.867416,2.722563,-5.082046,-1.863808,0.350806,9.073332,6.944360,-7.174525,-3.482676,8.099947,6.850501,-1.739196,3.069769,1.678253,-3.834052,-5.133764,-5.225891,-6.535392,2.353323,0.573235,-4.850464,-9.944039,-8.870240,-1.713343,-6.549936,5.652574,9.117303,9.533378,-6.859820,-6.870894,-7.773138,7.837127,0.583843,-5.383516,-3.359653,0.943517,-1.971645,-4.466321,-9.897227,-8.772445,4.882148,5.883819,4.380761,3.946517,1.277480,-2.668409,1.545740,-6.812919,9.343831,4.435189,-0.806754,-9.517780,4.099279,4.811515,5.288014,-0.064083,7.461498,-3.250741,-9.094158,9.348584,8.486172,1.810654,6.852137,9.591321,-0.554808,-5.279140,-9.223848,-2.299888,0.902409,9.858517,9.344180,8.360187,6.125455,0.036968,-9.095071,-0.568408,4.933061,-5.082046,7.469253,2.811898,-0.027940,8.254039,4.089801,-2.281555,7.854884,7.628792,7.564152,-0.532588,-0.099731,-7.866854,-9.366211,1.443411,5.850341,-8.500222,3.472450,-9.500065,1.441829,-6.152848,7.070680,1.045802,6.143262,2.841017,-0.976080,6.657367,-0.402857,2.346470,-4.231940,-9.092949,-9.661499,6.493271,-3.521559,5.246275,5.907605,-2.880072,0.794480,6.533645,7.287907,3.235544,-8.674046,-6.418540,2.555805,5.848711,-9.308406,8.975197,-1.591153,8.851458,-8.107559,-9.726973,-4.462123,7.238084,9.229258,-4.598554,0.301352,-6.783179,0.132632,-6.530959,-3.683080,-6.418361,8.028275,9.764388,-9.324098,1.438955,6.562562,-3.583500,-4.832868,1.440439,4.272769,-8.289433,-5.841135,-0.992066,-3.988685,-4.682518,2.403591,7.438988,-6.112273,-6.981014,9.926227,1.866132,-9.178881,5.783181,1.003313,-9.424199,-8.221034,-0.080795,6.150254,-7.580023,3.874290,-6.174464,-2.776984,-7.006967,-1.642696,1.415360,-9.628699,-0.884583,3.211640,1.445720,0.258244,5.196174,4.645469,-9.447782,-7.187648,-3.988371,6.037234,-5.283886,-5.384661,-5.000633,-1.395191,7.256893,-8.275001,7.115872,7.404085,8.286478,-8.435161,-1.302527,-1.713615,-3.702036,-0.066011,-4.599956,4.550082,9.985157,5.694802,-6.350649,8.629589,-8.451995,6.131640,4.746006,2.596582,4.855955,5.540046,4.884187,-8.824144,0.385896,-2.804176,-5.355410,-6.415109,4.197254,5.617326,4.444193,5.910038,3.117517,4.600318,3.905112,8.674599,-7.337948,8.673079,-2.090713,1.062689,-3.128705,5.170638,8.171234,-8.434358,-2.721367,-5.173199,-1.903373,3.338069,3.475136,4.940766,0.726810,8.003020,1.511569,0.937336,1.464534,0.927910,-3.516501,-9.766272,3.348383,3.539844,5.709385,5.881130,4.742652,4.944484,-5.955405,-9.582467,1.778959,2.873190,-8.848859,-8.478922,-5.709260,8.813276,-8.225290,-5.644930,-3.348857,6.986273,-8.491459,5.509093,-6.010541,-0.852554,6.348615,-4.014555,-2.765985,-2.718159,8.308039,-9.822993,-7.883496,-3.419903,-6.407356,-3.141236,-3.879933,-9.387357,8.754241,0.524930,-2.003810,-7.235496,-1.505668,-1.431763,-1.443820,1.366353,1.467717,2.894835,5.970533,-8.723640,0.891779,-6.149965,4.534176,-2.018066,-8.946351,-7.637736,4.984401,4.833755,8.722971,-6.999108,-1.754467,4.962798,-1.033701,4.393167,6.627260,5.398519,8.709613,-4.111772,-9.012433,6.889248,-6.916704,7.786140,-2.693816,1.457317,-7.478202,0.048231,2.975134,-1.404412,-8.628913,7.010025,3.252546,-7.045860,-2.369045,0.949468,5.637701,-2.082159,-2.229021,4.859496,-2.594759,6.944859,-3.737383,6.930539,1.590762,6.848848,1.474097,4.473928,9.693169,3.977073,6.532406,-4.615355,5.958267,-5.433004,-5.312749,-0.609160,3.062992,5.778910,8.242970,5.287337,-2.090475,-4.844920,-0.323521,9.460365,0.713675,3.000271,1.058340,2.773103,-7.029175,0.835206,9.352172,8.475677,4.426572,-5.014749,2.271846,0.484616,7.834650,-9.345563,-7.181512,7.945313,-0.003468,-2.121649,0.177463,9.229361,4.353651,-2.612609,-3.050413,8.330254,4.986510,-5.700599,-9.976052,-8.544956,4.273499,3.943481,-8.183925,6.750475,5.500339,4.098899,9.909892,2.193790,4.931767,8.469737,5.525113,1.930646,2.359091,-1.071641,7.814123,2.836166,3.605570,-7.174911,8.413401,-8.387830,4.229023,9.921124,-8.823110,-5.935330,7.872119,-5.230394,-1.147660,5.004656,7.472862,9.642015,-0.990719,-3.027258,9.108062,-0.346203,-5.924266,-4.348187,-2.074362,1.221184,-9.576188,-5.215389,8.909931,-0.812021,-4.632578,6.781390,5.931596,-7.647636,4.258720,1.277424,-6.643590,-1.179442,0.552225,-9.146442,-2.279790,-4.236863,6.880544,-2.454466,4.608161,9.635912,-9.691665,-3.912285,6.885058,-4.621891,0.983224,-4.675564,5.826990,-4.231512,1.403505,-4.333129,-2.766748,-8.730782,1.974007,4.192972,0.493166,-7.856409,3.697818,-5.355853,7.695382], dtype = "float32")#candidate|12585|(792,)|const|float32
var_12586 = relay.var("var_12586", dtype = "float64", shape = (32, 40))#candidate|12586|(32, 40)|var|float64
call_12584 = relay.TupleGetItem(func_9004_call(relay.reshape(const_12585.astype('float32'), [792,]), relay.reshape(var_12586.astype('float64'), [1280,]), ), 2)
call_12587 = relay.TupleGetItem(func_9008_call(relay.reshape(const_12585.astype('float32'), [792,]), relay.reshape(var_12586.astype('float64'), [1280,]), ), 2)
output = relay.Tuple([call_12562,call_12581,call_12584,const_12585,var_12586,])
output2 = relay.Tuple([call_12563,call_12582,call_12587,const_12585,var_12586,])
func_12593 = relay.Function([var_12586,], output)
mod['func_12593'] = func_12593
mod = relay.transform.InferType()(mod)
mutated_mod['func_12593'] = func_12593
mutated_mod = relay.transform.InferType()(mutated_mod)
var_12594 = relay.var("var_12594", dtype = "float64", shape = (32, 40))#candidate|12594|(32, 40)|var|float64
func_12593_call = mutated_mod.get_global_var('func_12593')
call_12595 = func_12593_call(var_12594)
output = call_12595
func_12596 = relay.Function([var_12594], output)
mutated_mod['func_12596'] = func_12596
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6880_call = mod.get_global_var('func_6880')
func_6881_call = mutated_mod.get_global_var('func_6881')
call_12623 = relay.TupleGetItem(func_6880_call(), 0)
call_12624 = relay.TupleGetItem(func_6881_call(), 0)
func_3278_call = mod.get_global_var('func_3278')
func_3282_call = mutated_mod.get_global_var('func_3282')
const_12628 = relay.const([-8,7,3,-3,5,8,-7,6,4,2,-3,10,7,-5,10,7,5,9,9,8,1,6,3,-8,1,-7,-8,-8,2,1,-1,2,-2,3,-3,5,-7,4,3,8,-3,-10,-2,5,10,-10,6,-7,2,-10,-3,9,-6,-9,-4,-2,-4,7,9,-3,-7,1,2,-2,2,-9,8,2,-10,4,-9,-10,-9,2,8,10,-6,-7,-3,-5,2,7,-2,8,1,6,10,-5,-3,-10,2,-4,-3,1,3,-8,-9,-7,-3,3,-8,-5,4,-9,9,-8,9,1,7,5,-7,-7,7,9,10,5,-10,-6,10,-3,-5,10,-5,6,-5,3,4,-10,3,-3,1,1,-8,-6,-4,-7,-2,-5,-10,-10,-1,-6,-1,8,-4,-9,8,-6,8,-7,3,-1,1,-1,-6,5,-3,4,7,-10,-4,-6,8,1,6,-8,9,5,5,2,-8,-1,-8,-5,-9,-4,1,-5,3,-4,-3,-9,2,-4,-1,-10,-1,-10,-10,-2,9,1,-9,10,4,3,4,10,8,-3,-8,8,8,10,2,-6,-9,2,3,2,-3,5,-8,3,6,-3,-1,-6,8,6,8,-10,6,9,-3,9,5,-5,-6,7,6,-8,2,7,-2,1,-4,-8,-4,-2,-7,2,4,5,6,3,3,-1,7,-6,-1,-10,-1,-5,-5,-3,-9,-4,-4,4,10,-4,-1,9,-10,-9,-5,6,-9,-10,-8,-9,-8,10,2,2,3,7,8,6,-3,1,-5,-2,7,-9,5,-4,10,4,9,-6,-9,-10,8,-9,10,7,-8,-2,-10,-4,6,7,3,10,7,-8,-6,7,2,-5,7,-2,10,4,2,7,5,5,-5,-3,10,1,8,10,-4,-4,6,8,9,-5,9,10,6,-3], dtype = "uint16")#candidate|12628|(336,)|const|uint16
call_12627 = func_3278_call(relay.reshape(const_12628.astype('uint16'), [6, 8, 7]), relay.reshape(const_12628.astype('uint16'), [6, 8, 7]), )
call_12629 = func_3278_call(relay.reshape(const_12628.astype('uint16'), [6, 8, 7]), relay.reshape(const_12628.astype('uint16'), [6, 8, 7]), )
output = relay.Tuple([call_12623,call_12627,const_12628,])
output2 = relay.Tuple([call_12624,call_12629,const_12628,])
func_12640 = relay.Function([], output)
mod['func_12640'] = func_12640
mod = relay.transform.InferType()(mod)
output = func_12640()
func_12641 = relay.Function([], output)
mutated_mod['func_12641'] = func_12641
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12067_call = mod.get_global_var('func_12067')
func_12069_call = mutated_mod.get_global_var('func_12069')
call_12700 = relay.TupleGetItem(func_12067_call(), 0)
call_12701 = relay.TupleGetItem(func_12069_call(), 0)
func_11803_call = mod.get_global_var('func_11803')
func_11806_call = mutated_mod.get_global_var('func_11806')
const_12720 = relay.const([[True,False,True,False,True,True,False,True,False],[True,False,False,True,False,False,True,False,True],[True,True,False,False,False,True,False,False,True],[False,False,False,True,True,True,False,True,False],[True,False,False,True,True,False,False,False,False],[False,True,False,True,True,True,True,True,True],[False,True,True,False,True,True,False,False,True],[True,True,False,False,False,False,False,False,False],[False,False,False,False,False,False,False,False,True],[False,True,False,True,False,True,True,False,False],[False,False,False,True,False,True,False,False,False],[False,False,False,False,False,False,True,False,False],[True,False,False,False,False,False,False,False,False],[True,True,False,True,True,False,True,True,False],[True,True,False,True,True,False,True,False,False],[True,True,False,True,False,True,False,False,True],[False,False,True,False,True,False,False,False,False],[False,True,True,True,False,False,True,True,True],[False,False,True,False,True,False,True,True,False],[False,True,False,True,True,False,True,True,True],[True,True,True,True,True,True,False,False,True],[False,False,True,False,False,True,True,False,True],[False,True,False,False,True,True,True,True,False],[True,False,True,True,True,False,False,True,True],[True,True,False,True,False,False,False,False,False],[True,True,False,False,True,False,False,False,True],[True,True,False,True,False,False,False,True,False],[False,False,True,False,True,False,False,False,False],[False,False,False,False,False,False,False,False,True],[False,True,True,False,False,False,False,False,True],[True,False,False,True,True,False,False,True,False],[False,True,False,True,True,True,True,True,True],[True,True,True,True,False,False,False,False,True],[True,False,True,False,True,True,False,False,False],[True,False,True,True,True,False,True,False,True],[False,True,False,True,True,False,True,False,False],[True,False,False,False,True,False,True,True,False],[True,False,False,True,True,True,True,False,False],[True,True,False,True,False,True,True,True,False],[True,False,False,True,False,True,True,False,False],[False,False,False,False,True,False,False,False,True],[True,True,True,True,True,False,True,False,False],[True,True,False,True,True,False,False,False,False],[True,False,True,True,False,True,True,False,False],[True,False,False,False,False,False,True,True,True],[False,True,True,False,False,False,False,True,True],[True,True,False,True,False,False,False,False,False],[True,True,False,True,True,False,False,True,False],[False,True,False,True,True,True,False,True,True],[False,False,False,False,True,False,False,False,False],[False,True,False,False,False,False,False,False,True],[False,False,False,True,False,True,True,True,False],[False,True,False,True,True,False,True,True,True],[True,True,True,False,True,False,False,True,True],[False,True,True,True,False,True,False,True,True],[True,False,False,False,False,True,False,True,False],[False,False,False,True,False,True,False,False,True],[True,False,False,False,True,False,True,False,False],[False,False,True,True,False,True,True,False,True],[False,False,False,False,False,False,False,True,False],[False,True,False,True,False,True,True,False,False],[True,True,False,True,True,True,False,False,True],[True,True,True,False,True,True,False,False,True],[True,False,False,True,False,True,False,True,False],[True,True,False,True,False,False,True,False,True],[False,False,False,True,False,True,False,False,True],[True,True,True,False,False,True,False,True,False],[False,True,True,False,False,False,False,True,True],[False,True,True,False,True,True,False,True,True],[True,True,False,False,True,True,True,True,True],[True,False,False,True,False,False,False,False,False],[True,False,False,False,True,True,True,True,False],[False,True,True,False,False,True,False,True,False],[True,False,True,False,True,True,False,False,True],[False,True,True,True,False,False,True,False,True],[False,True,False,False,True,True,False,True,True],[True,True,False,True,False,True,False,False,False],[False,True,True,True,True,False,True,False,False],[True,True,False,False,True,True,False,False,False],[True,False,False,False,True,True,True,False,False],[False,False,False,False,False,True,True,False,True],[True,True,True,True,False,True,True,False,False],[False,False,False,True,True,False,True,True,True],[False,False,True,False,False,False,True,True,False],[False,False,False,True,False,True,True,True,True],[False,True,True,False,True,True,True,False,False],[True,True,False,True,False,False,False,False,True],[False,True,False,True,False,False,True,True,False],[True,False,False,False,False,True,False,True,False],[True,False,True,False,True,False,False,False,False],[False,True,True,True,True,False,True,False,False],[False,False,False,False,True,False,False,True,True],[True,True,False,False,True,False,True,True,True],[True,True,False,False,True,True,True,True,False],[True,False,True,False,False,True,False,True,False],[False,True,True,True,True,True,True,True,False],[True,False,True,True,True,False,True,True,True],[False,False,False,False,False,True,True,False,True],[False,False,False,False,False,False,False,False,True],[True,True,False,True,True,False,False,False,True],[False,True,True,True,True,True,False,True,True],[True,True,True,False,True,True,True,False,False],[True,False,False,False,True,True,False,False,False],[False,False,False,True,False,True,False,True,False],[False,True,False,True,False,True,True,True,False]], dtype = "bool")#candidate|12720|(105, 9)|const|bool
call_12719 = relay.TupleGetItem(func_11803_call(relay.reshape(const_12720.astype('bool'), [945,])), 4)
call_12721 = relay.TupleGetItem(func_11806_call(relay.reshape(const_12720.astype('bool'), [945,])), 4)
output = relay.Tuple([call_12700,call_12719,const_12720,])
output2 = relay.Tuple([call_12701,call_12721,const_12720,])
func_12725 = relay.Function([], output)
mod['func_12725'] = func_12725
mod = relay.transform.InferType()(mod)
mutated_mod['func_12725'] = func_12725
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12725_call = mutated_mod.get_global_var('func_12725')
call_12726 = func_12725_call()
output = call_12726
func_12727 = relay.Function([], output)
mutated_mod['func_12727'] = func_12727
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7737_call = mod.get_global_var('func_7737')
func_7738_call = mutated_mod.get_global_var('func_7738')
call_12731 = func_7737_call()
call_12732 = func_7737_call()
output = call_12731
output2 = call_12732
func_12751 = relay.Function([], output)
mod['func_12751'] = func_12751
mod = relay.transform.InferType()(mod)
mutated_mod['func_12751'] = func_12751
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12751_call = mutated_mod.get_global_var('func_12751')
call_12752 = func_12751_call()
output = call_12752
func_12753 = relay.Function([], output)
mutated_mod['func_12753'] = func_12753
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11447_call = mod.get_global_var('func_11447')
func_11449_call = mutated_mod.get_global_var('func_11449')
call_12772 = func_11447_call()
call_12773 = func_11447_call()
output = call_12772
output2 = call_12773
func_12778 = relay.Function([], output)
mod['func_12778'] = func_12778
mod = relay.transform.InferType()(mod)
output = func_12778()
func_12779 = relay.Function([], output)
mutated_mod['func_12779'] = func_12779
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11055_call = mod.get_global_var('func_11055')
func_11057_call = mutated_mod.get_global_var('func_11057')
call_12805 = relay.TupleGetItem(func_11055_call(), 5)
call_12806 = relay.TupleGetItem(func_11057_call(), 5)
func_7656_call = mod.get_global_var('func_7656')
func_7657_call = mutated_mod.get_global_var('func_7657')
call_12811 = relay.TupleGetItem(func_7656_call(), 1)
call_12812 = relay.TupleGetItem(func_7657_call(), 1)
func_10703_call = mod.get_global_var('func_10703')
func_10705_call = mutated_mod.get_global_var('func_10705')
call_12817 = relay.TupleGetItem(func_10703_call(), 0)
call_12818 = relay.TupleGetItem(func_10705_call(), 0)
output = relay.Tuple([call_12805,call_12811,call_12817,])
output2 = relay.Tuple([call_12806,call_12812,call_12818,])
func_12820 = relay.Function([], output)
mod['func_12820'] = func_12820
mod = relay.transform.InferType()(mod)
output = func_12820()
func_12821 = relay.Function([], output)
mutated_mod['func_12821'] = func_12821
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5182_call = mod.get_global_var('func_5182')
func_5184_call = mutated_mod.get_global_var('func_5184')
call_12838 = func_5182_call()
call_12839 = func_5182_call()
output = relay.Tuple([call_12838,])
output2 = relay.Tuple([call_12839,])
func_12843 = relay.Function([], output)
mod['func_12843'] = func_12843
mod = relay.transform.InferType()(mod)
output = func_12843()
func_12844 = relay.Function([], output)
mutated_mod['func_12844'] = func_12844
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10179_call = mod.get_global_var('func_10179')
func_10180_call = mutated_mod.get_global_var('func_10180')
call_12888 = func_10179_call()
call_12889 = func_10179_call()
func_8031_call = mod.get_global_var('func_8031')
func_8033_call = mutated_mod.get_global_var('func_8033')
var_12925 = relay.var("var_12925", dtype = "float64", shape = (5, 132))#candidate|12925|(5, 132)|var|float64
call_12924 = relay.TupleGetItem(func_8031_call(relay.reshape(var_12925.astype('float64'), [660,])), 0)
call_12926 = relay.TupleGetItem(func_8033_call(relay.reshape(var_12925.astype('float64'), [660,])), 0)
output = relay.Tuple([call_12888,call_12924,var_12925,])
output2 = relay.Tuple([call_12889,call_12926,var_12925,])
func_12927 = relay.Function([var_12925,], output)
mod['func_12927'] = func_12927
mod = relay.transform.InferType()(mod)
var_12928 = relay.var("var_12928", dtype = "float64", shape = (5, 132))#candidate|12928|(5, 132)|var|float64
output = func_12927(var_12928)
func_12929 = relay.Function([var_12928], output)
mutated_mod['func_12929'] = func_12929
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5332_call = mod.get_global_var('func_5332')
func_5333_call = mutated_mod.get_global_var('func_5333')
call_12945 = relay.TupleGetItem(func_5332_call(), 1)
call_12946 = relay.TupleGetItem(func_5333_call(), 1)
output = call_12945
output2 = call_12946
func_12992 = relay.Function([], output)
mod['func_12992'] = func_12992
mod = relay.transform.InferType()(mod)
output = func_12992()
func_12993 = relay.Function([], output)
mutated_mod['func_12993'] = func_12993
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8623_call = mod.get_global_var('func_8623')
func_8625_call = mutated_mod.get_global_var('func_8625')
call_13004 = relay.TupleGetItem(func_8623_call(), 1)
call_13005 = relay.TupleGetItem(func_8625_call(), 1)
func_8388_call = mod.get_global_var('func_8388')
func_8390_call = mutated_mod.get_global_var('func_8390')
call_13013 = relay.TupleGetItem(func_8388_call(), 2)
call_13014 = relay.TupleGetItem(func_8390_call(), 2)
output = relay.Tuple([call_13004,call_13013,])
output2 = relay.Tuple([call_13005,call_13014,])
func_13045 = relay.Function([], output)
mod['func_13045'] = func_13045
mod = relay.transform.InferType()(mod)
output = func_13045()
func_13046 = relay.Function([], output)
mutated_mod['func_13046'] = func_13046
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8235_call = mod.get_global_var('func_8235')
func_8237_call = mutated_mod.get_global_var('func_8237')
call_13047 = func_8235_call()
call_13048 = func_8235_call()
output = call_13047
output2 = call_13048
func_13054 = relay.Function([], output)
mod['func_13054'] = func_13054
mod = relay.transform.InferType()(mod)
mutated_mod['func_13054'] = func_13054
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13054_call = mutated_mod.get_global_var('func_13054')
call_13055 = func_13054_call()
output = call_13055
func_13056 = relay.Function([], output)
mutated_mod['func_13056'] = func_13056
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11412_call = mod.get_global_var('func_11412')
func_11413_call = mutated_mod.get_global_var('func_11413')
call_13060 = relay.TupleGetItem(func_11412_call(), 0)
call_13061 = relay.TupleGetItem(func_11413_call(), 0)
output = call_13060
output2 = call_13061
func_13062 = relay.Function([], output)
mod['func_13062'] = func_13062
mod = relay.transform.InferType()(mod)
output = func_13062()
func_13063 = relay.Function([], output)
mutated_mod['func_13063'] = func_13063
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11055_call = mod.get_global_var('func_11055')
func_11057_call = mutated_mod.get_global_var('func_11057')
call_13113 = relay.TupleGetItem(func_11055_call(), 1)
call_13114 = relay.TupleGetItem(func_11057_call(), 1)
output = relay.Tuple([call_13113,])
output2 = relay.Tuple([call_13114,])
func_13116 = relay.Function([], output)
mod['func_13116'] = func_13116
mod = relay.transform.InferType()(mod)
mutated_mod['func_13116'] = func_13116
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13116_call = mutated_mod.get_global_var('func_13116')
call_13117 = func_13116_call()
output = call_13117
func_13118 = relay.Function([], output)
mutated_mod['func_13118'] = func_13118
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9925_call = mod.get_global_var('func_9925')
func_9927_call = mutated_mod.get_global_var('func_9927')
call_13124 = relay.TupleGetItem(func_9925_call(), 0)
call_13125 = relay.TupleGetItem(func_9927_call(), 0)
output = relay.Tuple([call_13124,])
output2 = relay.Tuple([call_13125,])
func_13138 = relay.Function([], output)
mod['func_13138'] = func_13138
mod = relay.transform.InferType()(mod)
mutated_mod['func_13138'] = func_13138
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13138_call = mutated_mod.get_global_var('func_13138')
call_13139 = func_13138_call()
output = call_13139
func_13140 = relay.Function([], output)
mutated_mod['func_13140'] = func_13140
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12507_call = mod.get_global_var('func_12507')
func_12508_call = mutated_mod.get_global_var('func_12508')
call_13144 = func_12507_call()
call_13145 = func_12507_call()
func_9988_call = mod.get_global_var('func_9988')
func_9990_call = mutated_mod.get_global_var('func_9990')
const_13147 = relay.const([[4.469867],[7.821289],[6.619833],[3.431075],[6.625913],[4.160600],[-4.460837],[4.244336],[4.051052],[3.018162],[-8.320268],[-0.109558],[6.445552],[0.155421],[-3.736546],[-1.395824],[3.523311],[-8.899830],[-3.844198],[0.362328],[-8.239622],[6.162365],[-3.634753],[-9.356961],[-4.463419],[-8.820147],[-6.780743],[-3.561767],[-3.077606],[-6.284258],[3.061105],[-4.711178],[-9.424005],[3.416448],[-6.019864],[-1.408535],[6.799041],[3.516575],[-4.414929],[-2.628149],[6.553406],[2.255602],[3.000510],[9.290071],[-2.797821],[-0.423487],[7.895946],[-3.219136],[-8.604701],[-5.309667],[-2.882580],[2.727725],[-9.030848],[6.063000],[-4.545760],[-4.833223],[-8.300400],[-7.587004],[0.751556],[3.993732],[0.190066],[9.513633],[-0.497052],[-6.113981],[0.657438],[-7.484592],[-3.872782],[-6.191152],[-9.758840],[0.393224],[6.893819],[5.571259],[-3.579021],[8.699022],[0.604955],[9.678309],[-1.167386],[9.536488],[-9.934869],[7.610632],[9.824934],[-9.021360],[1.875177],[-0.077775],[-6.719403],[1.916868],[0.396313],[-9.253586],[-6.783415],[1.249376],[-0.263212],[-0.333451],[1.507407],[-9.791536],[7.202810],[-5.859167],[2.752112],[-6.454317],[2.278432],[-4.488032],[7.398813],[8.814924],[-5.931994],[-4.855890],[0.480674],[2.338887],[-3.931916],[-4.423078],[4.908662],[5.807252],[0.325652],[5.415791],[-7.833931],[0.812951],[9.847655],[-4.550258],[1.018508],[3.781444],[1.488449],[-1.910021],[-5.814951],[-0.070298],[5.794203],[-7.480516],[1.862828],[-9.416103],[4.219292],[1.796427],[6.812498],[1.951571],[2.890227],[9.553906],[0.011606],[-4.807476],[-3.406482],[8.484963],[-5.470337],[1.444030],[3.923324],[8.438809],[-1.805493],[7.403482],[8.596785],[7.417696],[3.338812],[4.383260],[-9.824998],[-7.749511],[-3.337890],[-7.814126],[5.009696],[2.658084],[-4.863414],[-3.890653],[-7.291148],[-5.673719],[2.489961],[3.228367],[3.540650],[3.678911],[8.970366],[0.412787],[-8.248647],[-7.374328],[-0.810720],[3.941622],[-9.081974],[-5.557570],[-5.393224],[-3.797797],[-8.595586],[3.005318],[-7.246923],[-8.500415],[5.326413],[7.048717],[3.160509],[9.118357],[-8.541893],[-3.280113],[-1.483287],[-5.964358],[-6.192017],[3.363064],[0.860895],[9.376057],[4.574948],[1.029730],[6.276655],[5.379219],[-3.423963],[-2.035408],[3.147486],[6.774751],[0.466017],[-1.236364],[1.073940],[-5.881143],[2.814922],[-4.025108],[-2.178694],[-5.700374],[-2.556507],[1.783816],[-7.140167],[8.288374],[4.312593],[-8.932100],[1.651429],[-4.382114],[2.096619],[-9.835655],[-7.007680],[-5.429617],[8.355461],[-1.658225],[-1.374001],[7.523476],[5.066630],[8.708235],[2.215140],[-2.671735],[-7.902737],[2.442152],[-1.573433],[-6.053094],[-9.581964],[-1.632410],[9.005395],[4.558363],[-9.146766],[-1.889479],[-1.527367],[5.383551],[4.703999],[6.565476],[5.581005],[9.649203],[-5.267857],[3.403295],[4.588961],[-2.979749],[5.508377],[-3.914151],[2.018376],[7.224915],[7.207671],[6.459429],[1.200540],[4.129885],[1.798298],[5.484104],[-1.085296],[-6.988135],[4.847918],[9.973028],[3.146134],[-3.382760],[4.696299],[0.522472],[-9.467637],[-8.953064],[9.878394],[-7.723753],[-2.827238],[8.625615],[-4.702304],[5.247702],[9.288781],[-4.307185],[0.575567],[8.456825],[0.309918],[7.707374],[-1.299903],[-1.956282],[9.269728],[3.210519],[-3.213189],[1.624560],[9.121132],[-7.048789],[2.541136],[2.005134],[9.383868],[0.882043],[0.014903],[5.437427],[7.996962],[3.638324],[5.617509],[-7.367321],[-7.189915],[3.347721],[8.372631],[-0.626803],[0.998551],[9.032743],[-8.510967],[-3.530409],[-1.485370],[-4.246624],[-0.431241],[-1.595282],[5.645210],[-9.420953],[7.981299],[1.321952],[-6.830875],[-0.188996],[-4.362050],[-1.842354],[0.976901],[-0.700172],[-7.949319],[0.529450],[7.194818],[0.762124],[8.975447],[3.964198],[6.943043],[-6.185510],[-9.861811],[0.856857],[-5.509495],[1.129681],[-0.852257],[6.359659],[1.920131],[9.763671],[-3.768390],[-9.988007],[-1.256964],[6.004493],[6.825548],[-7.176361],[-7.806315],[-8.417870],[9.223764],[9.859940],[7.269113],[-5.430088],[8.483111],[-3.220994],[5.832981],[-6.028357],[-4.826181],[8.179295],[-3.793838],[0.088007],[9.984831],[1.202669],[-7.562016],[3.265509],[4.822599],[7.521375],[2.287602],[-1.550305],[4.318195],[6.658594],[-5.368346],[-1.700920],[-2.228985],[5.997658],[6.494329],[-8.739242],[1.765537],[-6.360256],[8.669989],[5.155044],[1.166160],[-3.984585],[-0.458903],[2.550583],[9.954652],[-1.601780],[-6.708596],[-9.035906],[0.688128],[-6.605908],[-5.129837],[8.662439],[-5.746335],[2.167115],[3.556774],[-2.306848],[-6.193647],[2.978605],[3.308524],[-1.120726],[3.246192],[-4.202320],[-1.376898],[-1.705898],[-6.199331],[-5.710231],[-0.746424],[-1.308410],[6.384198],[-1.857725],[-5.661812],[1.945359],[6.880435],[7.767044],[3.954652],[8.368440],[2.694866],[1.204949],[7.505751],[1.861381],[-6.587806],[-7.845293],[3.909556],[3.264137],[0.708704],[-5.761926],[-3.668325],[-0.198733],[-7.413364],[-3.076840],[2.662097],[2.054179],[4.344117],[-7.043745],[-5.059546],[-0.370829],[8.635992],[2.674318],[-2.589915],[-1.621308],[3.625888],[3.537160],[-7.618937],[2.336514],[-8.999913],[3.217232],[-7.142704],[-5.654754],[2.113830],[-9.108462],[-9.327225],[-1.364033],[8.730564],[-8.221459],[-6.587628],[9.465355],[5.751649],[9.143825],[-2.878750],[3.674061],[-4.859208],[7.474341],[7.472045],[-8.760168],[-5.389034],[-3.668022],[3.260548],[6.962252],[3.705097],[-4.919740],[-8.706632],[8.641570],[4.113462],[7.845439],[2.164429],[-5.179246],[7.389483],[-9.333638],[9.700452],[4.328226],[0.140703],[-0.846402],[-4.877236],[-2.350559],[6.290951],[1.551610],[-6.934698],[-4.326111],[8.398310],[-5.842912],[-2.605339],[-0.089019],[-4.722192],[-9.626018],[9.465084],[4.733447],[2.166160],[-4.778731],[8.462300],[6.743209],[5.767980],[-2.152930],[6.438619],[6.493109],[5.249936],[-5.697358],[-1.555204],[5.512795],[7.867934],[-7.639505],[-0.538008],[-7.094623],[2.340885],[4.233732],[7.666987],[-1.541246],[-2.676359],[-5.838829],[8.819442],[-6.857607],[3.326990],[-7.546077],[-8.533481],[-5.992288],[6.940082],[-9.655896],[0.306807],[9.291284],[4.763358],[3.127537],[-3.207070],[-4.423100],[5.020069],[-6.547282],[-2.068933],[6.956592],[4.816367],[-5.652549],[8.452759],[-9.354002],[8.823697],[-6.674878],[6.879544],[-4.432278],[7.885754],[1.158470],[6.627528],[-0.461092],[-6.545036],[0.883706],[2.439047],[5.472078],[-2.248822],[-0.946094],[-0.419793],[-8.845436],[-0.955070],[-7.289293],[-6.253891],[-5.727889],[9.949063],[4.709482],[6.410862],[2.931692],[6.278200],[1.660377],[0.267952],[3.767372],[-3.784121],[0.263421],[-2.839887],[-0.442156],[6.871579],[-6.152606],[5.605161],[7.559571],[-2.472776],[4.833255],[-9.235655],[-3.289302],[1.908272],[-3.222595],[-6.097503],[-7.950222],[-4.453931],[5.032309],[9.597799],[8.597536],[3.691689],[1.401675],[-6.167123],[6.727373],[7.902697],[3.661054],[6.019933],[7.833737],[1.669479],[4.394637],[-5.873733],[-2.315736],[8.221600],[9.109065],[8.343774],[2.375493],[6.921709],[-1.023920],[3.711393],[-9.814422],[-5.243203],[-7.419449],[0.063128],[3.799931],[-4.469788],[-5.723413],[-4.301217],[6.306150],[7.513215],[3.077167],[8.198922],[5.497186],[9.946100],[-2.606314],[6.774783],[2.747728],[-7.369263],[6.011905],[-7.132735],[-7.426506],[4.310591],[7.286576],[5.031985],[0.591499],[5.402612],[-2.008811],[-7.374457],[4.326546],[2.483428],[-7.911869],[2.182157],[-6.970884],[-2.342729],[0.558458],[-8.197768],[0.700663],[5.403541],[-6.779790],[7.928813],[3.872828],[-3.333976],[6.781164],[6.236003],[-9.509034],[-8.873244],[1.939429],[-9.887273],[-7.018067],[4.116206],[9.459128],[0.095463],[8.828328],[3.045858],[-6.493405],[3.629292],[-7.773372],[5.565633],[-5.179633],[-6.103996],[6.310207],[-5.052252],[-3.632079]], dtype = "float64")#candidate|13147|(660, 1)|const|float64
call_13146 = relay.TupleGetItem(func_9988_call(relay.reshape(const_13147.astype('float64'), [660,])), 1)
call_13148 = relay.TupleGetItem(func_9990_call(relay.reshape(const_13147.astype('float64'), [660,])), 1)
output = relay.Tuple([call_13144,call_13146,const_13147,])
output2 = relay.Tuple([call_13145,call_13148,const_13147,])
func_13149 = relay.Function([], output)
mod['func_13149'] = func_13149
mod = relay.transform.InferType()(mod)
mutated_mod['func_13149'] = func_13149
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13149_call = mutated_mod.get_global_var('func_13149')
call_13150 = func_13149_call()
output = call_13150
func_13151 = relay.Function([], output)
mutated_mod['func_13151'] = func_13151
mutated_mod = relay.transform.InferType()(mutated_mod)
const_13219 = relay.const([[[5,1,-1,5,-5,7,-1,-10,8,-9,-5,-6,7,1],[2,-10,-1,-8,-5,8,4,-9,10,-4,-9,-1,-5,7],[10,9,-3,-10,-1,7,-3,-5,-10,-2,4,-9,2,-7],[6,10,-2,-1,-10,-9,9,-7,1,-1,-6,3,2,5],[3,-2,4,8,-3,-2,10,-1,9,5,-10,5,1,8],[-2,-7,4,8,3,5,5,-6,1,5,-5,-2,-7,3],[-2,1,-5,-1,-10,-9,4,-7,-1,3,-2,2,7,4],[-8,1,-9,8,-3,-5,7,-5,9,-2,-7,-4,3,8],[-9,-5,-1,-1,-9,6,-5,4,-6,4,3,6,1,10],[-8,-4,-3,6,-5,4,-3,7,-1,1,-10,-3,-3,6]],[[10,7,-6,-5,-2,-3,-5,3,9,-7,1,-10,-4,-5],[-1,-10,8,-9,8,3,4,-7,7,6,-9,8,2,2],[-10,-5,-9,2,-10,-2,-6,10,-5,6,6,7,6,5],[1,5,4,2,3,-1,-8,5,9,1,-10,-3,-2,4],[-6,-7,-6,-10,-10,-6,-8,-10,-1,10,8,-2,-10,-7],[-7,-10,9,4,10,10,6,-7,-2,-1,-1,2,2,-10],[-1,5,-5,5,-6,1,-10,-10,-6,3,3,2,-6,-5],[-8,-3,9,9,6,-10,-2,8,4,10,-4,-2,10,-3],[8,-5,1,7,7,-6,1,-5,10,7,-1,6,2,4],[-9,8,-5,6,-2,-5,-1,2,-6,-5,2,10,10,1]],[[-3,-4,4,-2,7,-4,-2,-8,10,5,8,-4,-3,-6],[-2,5,-3,-10,2,-5,-1,-8,8,6,-7,-5,2,-7],[-4,7,7,-1,-2,-9,-4,-3,-3,-8,10,-8,6,4],[-3,-2,-2,-4,1,-1,-8,-7,-5,9,-5,9,-2,2],[2,-7,-5,-3,10,-5,-3,-3,-7,5,-5,9,3,3],[5,-5,-1,2,-3,-2,2,-5,-6,-8,-1,4,-9,-5],[-1,10,-6,-2,9,-3,-9,-5,-2,6,1,2,8,9],[9,2,4,7,-6,9,5,8,-8,-4,5,10,-5,9],[8,-3,-4,8,-6,10,-4,-7,-2,2,10,-10,-6,-1],[3,-4,-10,8,1,-1,-1,-9,9,-5,-1,-1,-4,5]],[[1,1,-8,5,-4,-1,8,-2,-3,7,6,-5,10,-5],[-8,4,-6,5,2,3,2,-2,4,-1,-6,-10,-4,-9],[-10,-9,-7,-10,-10,9,5,-9,-4,7,-7,-6,1,-5],[-9,-9,1,9,-3,-10,-4,-8,-1,-5,-9,6,5,9],[7,10,10,-7,9,8,-5,-6,9,7,2,6,6,-7],[3,-4,-2,6,-6,-5,-9,7,-1,-2,-9,-10,-10,-10],[8,5,5,-10,6,1,7,6,-6,9,-8,-10,9,-7],[6,-5,-7,4,4,9,1,1,-3,-5,2,10,6,-6],[-8,6,-10,-8,3,-9,9,-8,-7,8,8,10,-7,9],[-10,8,5,4,5,2,-6,-9,-8,-5,6,6,-7,5]],[[9,-7,6,-9,-9,6,-4,6,-9,7,1,-4,-5,2],[-9,6,-10,8,-6,7,-9,1,-3,-7,9,-2,6,-9],[10,-3,-8,-8,-9,4,-3,-3,4,-8,-1,1,-7,-2],[3,-5,3,-2,6,9,6,3,4,2,9,7,2,-2],[3,5,-7,8,8,-1,2,8,-3,9,-6,-7,10,-9],[-2,-5,-6,2,1,-3,2,1,-4,-3,-1,-4,5,6],[-5,-2,-5,-3,-7,6,5,-9,3,-10,-1,4,7,-3],[-9,-8,1,1,-6,8,-7,3,4,-8,4,-4,3,8],[-7,-10,9,1,7,5,6,9,8,5,6,8,3,4],[9,-4,5,5,4,-4,-7,4,-4,9,5,-3,9,-10]],[[3,-8,7,1,8,6,1,4,-8,-3,3,6,6,-10],[1,9,3,-8,-9,-4,4,-2,-1,3,5,4,6,8],[9,8,6,-6,5,3,1,-3,-1,-10,-1,4,6,6],[7,-7,1,-10,5,10,-8,-3,4,5,-3,8,7,4],[9,9,-5,4,-2,-5,-10,6,-9,8,7,9,-10,5],[-6,-3,10,-10,-8,3,9,-6,4,-9,-1,6,-2,-9],[9,-2,-3,-9,4,-10,-4,-2,-10,3,2,-6,-9,-10],[-5,-5,-3,1,-1,10,10,8,2,8,-10,-5,5,5],[2,-6,6,-3,-10,6,-5,-7,-10,-8,8,-4,8,-3],[1,-8,-5,3,-8,2,-1,9,9,-2,2,-1,7,-1]],[[7,10,1,-2,-2,10,3,10,-2,-1,-5,-10,-2,-9],[8,-6,1,-2,3,1,-1,9,-2,-10,-8,-7,-4,1],[-10,7,-10,3,-8,-6,-3,10,7,-9,-8,5,-8,10],[1,-7,4,9,6,7,-8,-7,-1,4,6,-9,2,10],[-8,1,-4,2,-4,-3,-4,-6,-5,-3,10,4,-3,-2],[-9,-5,3,-8,-8,-4,-1,3,-7,-8,-2,9,-10,-9],[8,-8,-4,-3,10,-5,-2,-1,5,-8,2,7,7,-9],[-10,10,10,2,2,2,-9,-10,-7,3,6,-4,-9,4],[8,-5,-2,4,-6,7,-6,-9,9,3,10,-7,9,-7],[-4,-6,-4,9,-9,-9,3,-10,8,-1,9,-10,2,-7]],[[9,-1,5,-5,10,5,9,2,-2,1,4,-5,-8,3],[-4,-10,6,-6,-9,-10,-7,10,-10,9,7,-2,-2,3],[3,9,-10,-10,10,-7,-9,2,3,3,4,-5,-4,9],[5,-5,10,6,2,-6,2,-5,-10,8,-7,-9,6,2],[-3,5,3,-6,3,4,-8,10,-4,2,4,-1,2,1],[-6,-8,-4,-8,-1,3,-6,1,-7,-3,-8,7,4,2],[-8,6,8,4,6,-2,-8,-3,1,-8,5,-9,6,-6],[4,8,2,7,-9,5,-4,-2,-1,-5,-9,1,4,-8],[8,1,-1,-5,7,7,4,3,9,8,-2,-2,-8,4],[3,-2,-5,2,-4,-1,10,8,9,9,7,-8,2,10]],[[3,2,6,-8,7,-9,8,3,-9,1,2,-8,9,10],[5,3,-3,-5,-9,2,-2,8,-4,-9,3,2,2,-8],[3,-6,7,-8,4,-1,-6,-8,5,-10,8,3,3,-9],[-3,-8,5,9,-3,2,8,-8,-8,3,-6,-4,6,8],[-1,4,5,4,-6,-2,8,6,-1,2,-5,-3,-10,5],[-2,-3,3,-9,-9,-8,6,6,8,-1,8,-6,-10,4],[-4,6,-4,-1,8,5,4,2,3,-5,-10,10,2,1],[-5,-4,3,-6,-10,-6,-8,-4,2,2,6,8,6,5],[6,-1,-2,9,-4,-4,4,3,-9,-4,-4,10,-4,7],[-8,9,8,-2,-3,-6,10,3,10,-5,10,-1,6,4]],[[2,1,4,4,9,4,1,-1,4,4,-8,4,-10,4],[-8,7,2,2,9,-1,3,2,-7,-4,9,-4,7,-6],[-5,-3,3,-5,-2,-2,-2,-1,10,9,-8,-8,-1,3],[10,-6,-10,-10,-1,-9,-9,-5,-3,5,10,1,-9,5],[3,9,-7,-10,4,-9,2,-9,8,3,-10,7,4,-8],[-8,2,-10,5,-8,4,5,-8,-3,10,-2,-5,3,4],[9,-2,-4,-8,-9,-2,6,-8,9,-9,-4,-4,-4,-1],[-2,-8,-7,-7,7,5,-8,1,-3,6,7,-6,4,9],[2,7,-3,4,5,10,7,-8,-4,-1,-9,3,-2,-9],[1,-6,-6,-7,-6,-8,2,-3,-3,6,-1,6,-3,8]],[[-10,-4,4,3,1,-8,9,-9,4,-2,4,-6,-4,-9],[-3,-4,5,-3,-8,-9,-6,-9,3,-8,2,3,5,-5],[4,-7,2,-10,-3,-7,7,3,3,8,10,2,-5,1],[6,10,9,-7,-3,9,-9,-7,10,-2,2,2,8,-7],[1,-3,-2,-9,2,-2,-4,-1,7,1,-9,-2,-3,-4],[-8,-3,9,-8,-9,-1,-6,8,-6,-2,6,-10,-6,-8],[9,-9,-5,8,8,-4,-6,10,2,1,-4,10,-8,-8],[7,3,-10,-2,7,-8,9,-4,8,4,-9,-7,8,6],[8,-10,10,10,10,-4,6,-1,-5,-8,8,-1,9,-1],[-5,-4,-1,-9,-10,2,3,-6,-9,1,-4,4,-5,1]],[[5,2,1,5,10,2,3,9,-9,4,-6,9,-6,-3],[2,-2,6,-8,-9,8,7,9,-5,7,-7,-1,5,-4],[6,9,-1,-4,-6,5,-5,5,4,-5,6,2,-2,6],[-4,-6,10,-8,-6,-10,-5,8,-10,6,-6,-5,-3,-6],[10,2,-4,4,-4,5,-7,4,3,-5,-4,-9,2,-5],[-10,1,4,-6,4,2,-4,-9,5,-4,5,1,10,2],[-5,10,9,-5,-7,-9,3,-8,-10,10,2,4,-3,7],[-7,7,7,5,-9,6,3,6,1,-5,-2,-6,8,2],[-5,7,9,1,-9,3,-9,3,-6,-10,-2,-10,1,-9],[-6,-2,-1,10,-1,7,9,8,-8,-10,10,3,-6,-8]],[[-9,-5,7,5,-4,-7,5,-7,-3,2,7,-8,-3,-1],[-6,-5,-6,-1,6,3,10,-7,6,-9,8,-2,5,-2],[-3,-9,-1,-5,3,-4,-9,-10,-6,3,-10,1,-8,-7],[-5,3,-3,8,-5,3,-10,-8,3,-9,5,-3,6,7],[8,-7,2,-10,4,-10,-4,8,9,2,-4,-3,3,-7],[1,-5,9,-2,3,7,2,6,-10,10,10,3,3,6],[3,6,-9,-1,-2,1,-6,3,-8,6,5,9,-7,1],[10,5,3,5,2,-2,10,-2,-6,4,5,-9,-5,-8],[-3,-10,6,-6,9,-3,-7,7,-2,4,5,-9,-8,-1],[-9,5,7,4,-5,6,-2,-4,6,8,10,6,-10,6]]], dtype = "uint16")#candidate|13219|(13, 10, 14)|const|uint16
const_13220 = relay.const([[[-8,5,9,3,1,-1,5,-5,-8,-1,-2,-10,-5,-3],[9,-5,7,4,-6,-3,7,1,9,-8,8,6,8,-5],[8,-1,10,-5,-6,2,-7,10,-8,-9,4,3,-5,-4],[-2,5,6,10,-2,2,5,-7,-10,10,7,-6,5,6],[8,-4,-8,-4,-2,2,-1,-9,10,-1,-4,-10,-8,-5],[6,4,1,6,-4,-6,5,2,4,-2,-9,5,-2,8],[3,-5,-3,3,-7,-5,-9,9,1,-10,3,9,5,-7],[1,8,-7,1,4,-5,-6,1,-3,1,-2,-9,5,6],[2,-8,9,4,-1,-9,2,-3,6,4,-6,5,6,-2],[-2,2,4,-4,7,10,-7,-10,-2,-5,-1,-10,-4,3]],[[-9,-2,5,-4,3,1,-7,8,1,3,3,4,9,-10],[-8,-4,10,-5,-9,4,-8,3,-8,-9,1,4,10,-7],[-3,1,-2,6,10,-8,2,-4,-7,-6,-8,-5,5,-2],[-1,-3,7,-4,6,2,-5,6,2,-10,6,5,-10,-2],[2,-9,4,7,-5,-2,-3,6,7,-7,-1,6,-10,-1],[1,-5,-9,8,-8,5,-6,-4,-5,8,4,9,-2,-5],[5,-3,1,-1,-4,-2,-4,6,7,-10,2,-2,-3,4],[-7,-8,-4,1,10,-6,5,-9,-6,-10,6,-5,7,-9],[-10,8,3,-2,-8,4,6,-2,-7,-9,-5,6,-10,2],[-2,-10,5,2,5,-4,-6,8,-6,5,1,-6,5,-8]],[[7,-2,2,-1,-1,2,6,6,-4,-6,-7,6,6,-4],[6,-6,5,-9,-2,-3,-3,9,-3,-1,-1,7,-9,8],[5,-6,6,-3,2,6,-4,6,-4,-9,2,-3,-1,3],[-5,5,2,2,-5,-7,-9,-2,-3,-4,-3,-5,-5,8],[-9,5,-3,-4,4,-3,1,8,6,-4,-8,6,-1,6],[9,3,-6,-4,1,-4,-3,3,-8,5,3,-2,6,-2],[-1,7,4,-10,10,-9,-3,3,-1,-2,-9,2,-4,3],[-2,-1,5,3,-7,10,-8,-6,4,-8,10,-4,5,-3],[8,-1,2,3,9,7,-2,3,7,-10,-3,-4,8,-1],[3,4,-9,4,-6,-1,5,-2,6,-7,-7,4,7,2]],[[-7,10,5,1,-8,7,2,-10,-7,-3,8,-8,9,6],[1,3,8,-5,-7,7,1,-1,2,-9,-6,2,2,-10],[1,9,1,-9,7,-2,-2,4,-4,6,8,7,-6,7],[-8,-1,10,-2,-5,-10,6,-1,-4,2,1,6,-3,-3],[-1,-9,9,9,-1,2,3,-10,-5,-7,-5,-7,-9,6],[-5,1,3,-4,-1,2,5,9,6,-4,2,8,-4,-4],[7,5,2,5,-5,8,3,9,1,-5,-7,8,4,-7],[-2,-9,4,8,-9,3,-5,1,10,7,8,-10,2,-6],[-5,-5,7,-5,-8,-1,7,3,5,-1,6,10,-1,10],[-8,2,-3,-10,10,3,8,7,8,-9,-5,-9,10,6]],[[-10,7,-7,3,7,-10,9,-7,7,-4,5,9,-4,3],[-8,7,7,7,-4,-4,4,-8,-9,-1,6,-3,-1,-2],[-3,-10,4,-10,-3,5,-9,-8,2,7,-10,10,-6,-8],[-5,3,-3,-10,10,10,-9,1,-3,-3,7,-8,10,3],[-9,4,-1,-6,8,3,-8,-3,5,4,-5,5,-7,-6],[-9,5,-2,-5,-4,7,-7,-1,-1,-3,-10,-8,-6,-8],[-6,-1,5,-4,-3,-10,-2,-10,-5,-5,9,7,-6,6],[-8,6,9,8,3,1,-2,4,5,-4,-7,1,6,-5],[2,6,-6,-3,-10,-8,-4,-9,-9,-2,-7,7,5,9],[-5,-6,8,7,4,8,3,-7,5,3,8,-8,-2,-10]],[[2,-10,8,6,-10,7,-4,-8,-10,-4,-10,9,-4,5],[-2,7,-6,3,-3,-7,3,-6,10,-6,10,7,8,-10],[-1,-2,-9,8,-6,-9,-1,1,-4,-3,-5,-9,8,7],[-10,-2,-3,7,-3,-9,-6,-7,7,-3,9,7,6,8],[-10,1,3,4,-8,-8,-3,-10,-5,10,-7,3,10,9],[-4,9,4,-5,-5,-5,-7,6,1,-3,-4,-8,-5,-6],[-1,-10,-2,2,-10,6,-5,8,5,-5,9,3,-10,-4],[-4,-1,-3,-3,-9,4,-1,3,-1,-9,5,5,2,9],[10,1,4,-10,-1,4,7,7,8,3,3,8,-6,3],[-10,3,6,4,4,-4,9,6,6,2,-5,3,-9,-5]],[[-3,-4,-1,-5,-1,-4,4,-1,-8,7,-10,7,-5,3],[-3,-10,7,1,3,4,4,-10,-5,-9,9,8,6,5],[8,-7,8,-7,-3,2,4,-9,-3,-5,1,-1,10,10],[-7,-9,-4,-5,-6,-9,3,-3,9,-5,3,1,4,4],[3,1,-10,-4,9,-7,4,-1,9,-5,-10,-9,8,8],[-3,-7,-8,-6,-9,3,10,2,-7,-2,-9,-5,5,3],[-3,9,-5,2,-3,4,-7,-1,-8,-2,5,1,8,4],[-9,1,10,-10,4,-10,8,-10,-10,3,-3,-5,8,-4],[-2,-2,-7,-5,-10,-4,4,-3,8,1,-6,10,-1,10],[10,4,4,5,8,8,4,2,6,4,1,-5,-10,-8]],[[7,10,4,1,1,-1,3,-7,-1,-7,-4,-2,4,4],[2,7,-2,9,1,5,-4,-1,6,-8,-6,-6,5,9],[-5,5,5,-7,1,6,-5,-5,9,-10,3,4,2,-2],[-5,-6,4,-8,-5,7,-2,-7,-3,-8,8,7,-7,-3],[-10,5,5,-6,5,9,8,8,-10,8,9,8,-5,1],[8,2,8,-5,-3,4,-4,9,-1,-5,2,-6,9,-8],[-1,-8,-10,6,2,-3,-2,-2,-1,2,6,3,-9,8],[-3,2,-5,2,3,-5,9,9,-4,2,1,-7,1,6],[-9,8,1,5,-4,-1,-4,3,2,-7,-10,4,-2,8],[-1,3,-9,3,5,-3,-2,8,-9,-3,1,-1,-10,-2]],[[-8,-6,-5,10,-2,-3,8,7,-10,-9,-4,-2,-2,-7],[10,5,-2,-6,9,-2,-8,9,-10,-9,3,-1,-10,-3],[-8,5,10,5,-10,-10,8,2,-8,5,1,-10,1,1],[-4,6,-3,-9,2,6,6,2,9,-9,7,6,-7,-2],[6,2,8,-2,-10,5,-5,-3,-8,-5,9,5,-10,-5],[5,-7,-3,-3,-10,1,5,1,8,3,1,4,-3,-9],[9,-2,-3,9,10,6,-4,10,-2,1,7,5,-9,-1],[7,-5,-3,-5,6,4,-4,5,-7,9,-8,-9,-2,4],[-3,-6,-4,-1,10,5,9,-5,2,-3,7,-10,4,-9],[-6,-10,9,-1,10,9,-7,-9,8,1,4,-10,-8,3]],[[10,8,6,-6,5,7,1,-9,-4,7,2,3,2,6],[-4,1,2,1,-9,-4,-6,2,-1,-6,-9,9,8,9],[3,-1,-4,-2,-2,-6,-10,7,-3,3,-2,-1,9,4],[1,-9,-9,-3,5,-9,8,9,-10,-4,-5,-9,6,-10],[9,4,7,4,4,-9,3,-7,5,-9,-10,8,-1,-2],[6,-2,6,-4,-9,-9,6,-3,5,-1,6,5,-4,1],[9,-4,7,-1,4,-8,-6,8,-4,-2,-1,-9,-3,-2],[7,-6,-1,5,-6,3,-4,3,-7,-3,-3,9,-9,2],[-3,1,-10,-8,7,8,-10,3,1,1,8,8,10,-5],[6,-1,2,6,-10,10,8,1,9,9,2,8,5,-1]],[[-3,-3,2,5,5,3,-6,3,2,-1,1,-1,-1,3],[4,10,-10,4,7,5,1,-9,4,-2,-7,-7,2,-1],[4,-8,10,4,10,-2,-7,9,7,-6,2,-7,-2,7],[-2,-8,4,5,-3,4,8,8,-4,-5,7,-4,-1,9],[-7,9,7,-4,-1,-7,-9,-8,8,2,5,3,-9,2],[-9,-5,-9,4,2,-9,8,-7,-8,3,3,7,3,1],[-6,3,-5,-10,3,-10,-1,-9,-6,5,3,9,-1,5],[6,-7,-8,6,9,8,9,8,-7,-8,-5,-9,8,4],[4,-5,-5,3,-3,9,-4,-5,-4,-4,9,-6,-4,3],[-2,-5,-6,9,3,5,10,7,2,8,4,-9,-3,-7]],[[1,-10,5,-1,6,-2,-1,-6,-2,4,4,10,-3,2],[5,2,-7,8,6,5,-1,-2,6,1,10,2,7,-3],[-10,-6,-4,10,1,-5,9,-10,9,-3,9,-2,1,-2],[-7,-1,-1,1,-7,7,9,6,2,-8,-9,7,-5,6],[5,6,-4,9,4,-1,1,1,-1,-9,3,5,6,-10],[1,9,-4,7,2,5,6,-8,-5,-8,-4,-1,-4,-10],[8,-3,6,6,7,-7,-9,-4,10,3,-9,-2,-8,3],[-9,2,3,-6,-7,8,-6,-4,9,10,-10,-3,-6,5],[7,-3,8,-4,3,7,-4,6,10,-7,-8,8,-1,-10],[-7,-8,10,3,-9,9,8,-3,-3,-6,-1,-3,9,-8]],[[2,5,10,4,-9,9,3,-1,8,-7,5,5,6,4],[1,6,-7,-3,-8,-8,-3,4,-7,-8,9,5,1,-7],[-1,-2,2,-4,-1,-5,-3,4,9,3,-3,10,-8,-5],[6,6,4,8,6,-2,-10,-3,-9,8,-5,4,1,9],[1,-8,7,-4,2,4,5,-2,-4,-7,-2,-2,7,-6],[-2,-10,8,-9,4,5,-7,-1,-6,9,-9,-2,3,-9],[-2,1,-1,-6,-2,5,8,1,10,10,-6,5,-5,-8],[3,-1,2,8,6,4,6,-2,-9,2,7,-5,-8,8],[-7,6,-4,2,10,1,3,3,-5,6,8,-3,-9,-7],[9,2,4,6,4,5,-10,-1,-2,-4,9,-2,7,3]]], dtype = "uint16")#candidate|13220|(13, 10, 14)|const|uint16
bop_13221 = relay.subtract(const_13219.astype('uint16'), relay.reshape(const_13220.astype('uint16'), relay.shape_of(const_13219))) # shape=(13, 10, 14)
output = bop_13221
output2 = bop_13221
func_13227 = relay.Function([], output)
mod['func_13227'] = func_13227
mod = relay.transform.InferType()(mod)
output = func_13227()
func_13228 = relay.Function([], output)
mutated_mod['func_13228'] = func_13228
mutated_mod = relay.transform.InferType()(mutated_mod)
var_13323 = relay.var("var_13323", dtype = "uint8", shape = (12, 5, 6))#candidate|13323|(12, 5, 6)|var|uint8
var_13324 = relay.var("var_13324", dtype = "uint8", shape = (12, 5, 6))#candidate|13324|(12, 5, 6)|var|uint8
bop_13325 = relay.maximum(var_13323.astype('uint8'), relay.reshape(var_13324.astype('uint8'), relay.shape_of(var_13323))) # shape=(12, 5, 6)
uop_13330 = relay.asin(var_13323.astype('float32')) # shape=(12, 5, 6)
output = relay.Tuple([bop_13325,uop_13330,])
output2 = relay.Tuple([bop_13325,uop_13330,])
func_13347 = relay.Function([var_13323,var_13324,], output)
mod['func_13347'] = func_13347
mod = relay.transform.InferType()(mod)
mutated_mod['func_13347'] = func_13347
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13347_call = mutated_mod.get_global_var('func_13347')
var_13349 = relay.var("var_13349", dtype = "uint8", shape = (12, 5, 6))#candidate|13349|(12, 5, 6)|var|uint8
var_13350 = relay.var("var_13350", dtype = "uint8", shape = (12, 5, 6))#candidate|13350|(12, 5, 6)|var|uint8
call_13348 = func_13347_call(var_13349,var_13350,)
output = call_13348
func_13351 = relay.Function([var_13349,var_13350,], output)
mutated_mod['func_13351'] = func_13351
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9257_call = mod.get_global_var('func_9257')
func_9258_call = mutated_mod.get_global_var('func_9258')
call_13370 = relay.TupleGetItem(func_9257_call(), 0)
call_13371 = relay.TupleGetItem(func_9258_call(), 0)
output = relay.Tuple([call_13370,])
output2 = relay.Tuple([call_13371,])
func_13417 = relay.Function([], output)
mod['func_13417'] = func_13417
mod = relay.transform.InferType()(mod)
output = func_13417()
func_13418 = relay.Function([], output)
mutated_mod['func_13418'] = func_13418
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5143_call = mod.get_global_var('func_5143')
func_5144_call = mutated_mod.get_global_var('func_5144')
call_13441 = relay.TupleGetItem(func_5143_call(), 1)
call_13442 = relay.TupleGetItem(func_5144_call(), 1)
output = call_13441
output2 = call_13442
func_13453 = relay.Function([], output)
mod['func_13453'] = func_13453
mod = relay.transform.InferType()(mod)
mutated_mod['func_13453'] = func_13453
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13453_call = mutated_mod.get_global_var('func_13453')
call_13454 = func_13453_call()
output = call_13454
func_13455 = relay.Function([], output)
mutated_mod['func_13455'] = func_13455
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6875_call = mod.get_global_var('func_6875')
func_6876_call = mutated_mod.get_global_var('func_6876')
call_13493 = func_6875_call()
call_13494 = func_6875_call()
output = call_13493
output2 = call_13494
func_13519 = relay.Function([], output)
mod['func_13519'] = func_13519
mod = relay.transform.InferType()(mod)
mutated_mod['func_13519'] = func_13519
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13519_call = mutated_mod.get_global_var('func_13519')
call_13520 = func_13519_call()
output = call_13520
func_13521 = relay.Function([], output)
mutated_mod['func_13521'] = func_13521
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12725_call = mod.get_global_var('func_12725')
func_12727_call = mutated_mod.get_global_var('func_12727')
call_13526 = relay.TupleGetItem(func_12725_call(), 2)
call_13527 = relay.TupleGetItem(func_12727_call(), 2)
func_12725_call = mod.get_global_var('func_12725')
func_12727_call = mutated_mod.get_global_var('func_12727')
call_13528 = relay.TupleGetItem(func_12725_call(), 1)
call_13529 = relay.TupleGetItem(func_12727_call(), 1)
output = relay.Tuple([call_13526,call_13528,])
output2 = relay.Tuple([call_13527,call_13529,])
func_13534 = relay.Function([], output)
mod['func_13534'] = func_13534
mod = relay.transform.InferType()(mod)
mutated_mod['func_13534'] = func_13534
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13534_call = mutated_mod.get_global_var('func_13534')
call_13535 = func_13534_call()
output = call_13535
func_13536 = relay.Function([], output)
mutated_mod['func_13536'] = func_13536
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4857_call = mod.get_global_var('func_4857')
func_4858_call = mutated_mod.get_global_var('func_4858')
call_13551 = relay.TupleGetItem(func_4857_call(), 1)
call_13552 = relay.TupleGetItem(func_4858_call(), 1)
func_6875_call = mod.get_global_var('func_6875')
func_6876_call = mutated_mod.get_global_var('func_6876')
call_13565 = func_6875_call()
call_13566 = func_6875_call()
output = relay.Tuple([call_13551,call_13565,])
output2 = relay.Tuple([call_13552,call_13566,])
func_13569 = relay.Function([], output)
mod['func_13569'] = func_13569
mod = relay.transform.InferType()(mod)
mutated_mod['func_13569'] = func_13569
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13569_call = mutated_mod.get_global_var('func_13569')
call_13570 = func_13569_call()
output = call_13570
func_13571 = relay.Function([], output)
mutated_mod['func_13571'] = func_13571
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13534_call = mod.get_global_var('func_13534')
func_13536_call = mutated_mod.get_global_var('func_13536')
call_13632 = relay.TupleGetItem(func_13534_call(), 0)
call_13633 = relay.TupleGetItem(func_13536_call(), 0)
func_13116_call = mod.get_global_var('func_13116')
func_13118_call = mutated_mod.get_global_var('func_13118')
call_13650 = relay.TupleGetItem(func_13116_call(), 0)
call_13651 = relay.TupleGetItem(func_13118_call(), 0)
func_7737_call = mod.get_global_var('func_7737')
func_7738_call = mutated_mod.get_global_var('func_7738')
call_13680 = func_7737_call()
call_13681 = func_7737_call()
var_13682 = relay.var("var_13682", dtype = "bool", shape = (105, 9))#candidate|13682|(105, 9)|var|bool
bop_13683 = relay.divide(call_13632.astype('float64'), relay.reshape(var_13682.astype('float64'), relay.shape_of(call_13632))) # shape=(105, 9)
bop_13686 = relay.divide(call_13633.astype('float64'), relay.reshape(var_13682.astype('float64'), relay.shape_of(call_13633))) # shape=(105, 9)
uop_13690 = relay.log2(call_13632.astype('float64')) # shape=(105, 9)
uop_13692 = relay.log2(call_13633.astype('float64')) # shape=(105, 9)
func_8589_call = mod.get_global_var('func_8589')
func_8591_call = mutated_mod.get_global_var('func_8591')
const_13700 = relay.const([7.846879,3.522437,-1.126829,-6.007291,8.844773,4.412153,1.266244,9.601442,5.033647,5.742541,3.239170,1.908407,4.227320,8.765414,-3.556747,8.789445,-3.495167,8.967699,4.042543,-7.553876,-7.100627,0.405254,4.020127,-4.511141,-4.884819,3.987669,1.026108,9.607226,-5.455776,3.819783,3.299107,-1.204170,4.913337,7.172666,0.930073,3.892760,7.612912,0.162669,-6.594193,8.786359,-2.861058,-4.463443,3.181936,-5.056045,-0.198995,-5.190547,-3.878909,-5.362839,-3.598891,-9.827439,0.321920,-6.811576,-3.736359,3.756581,1.653795,-7.402142,6.095969,3.576558,-6.923655,-9.663745,1.057724,-0.442392,1.551199,-9.626763,-9.947819,-4.565465,-1.106407,-3.017366,-8.132916,2.803326,-1.251343,3.886888,6.970103,-9.977655,-3.217708,-8.669688,0.686564,1.713626,5.140507,3.033607,-6.314896,3.187070,-9.321458,-9.373739,-9.053264,-5.967950,0.816700,-7.548713,8.460701,-4.085838,-2.663832,1.224724,0.934814,-9.381647,-7.442180,3.758733,3.573559,-1.967926,-7.500600,3.976450,-1.110203,-2.942073,2.887227,-3.695834,9.431928,9.000801,-1.783513,-4.990306,8.536820,-8.463681,-5.806788,9.320310,7.161450,5.472576,-0.844752,-4.660823,-7.525918,-9.422930,2.187341,-1.830854,2.642344,-6.180446,0.175603,3.741182,-0.937477,-4.826606,0.524363,-5.502086,9.338634,3.940804,-1.538055,2.079348,-3.759856,1.756767,-4.893809,-1.469271,0.502006,2.252443,5.106675,9.929361,0.221095,-7.759691,9.648799,2.865794,-2.685938,-7.536069,-7.855236,-7.596279,5.363938,-6.384365,9.380808,6.771204,-2.843580,8.351860,-6.318387,0.371196,0.941987,-4.892099,-9.322585,9.312826,-0.686400,-0.504942,8.622374,3.207888,-9.638464,2.448234,3.125841,-3.935543,9.707268,7.095023,-0.848639,7.656996,1.249888,8.955128,5.186741,-9.887534,-0.821116,-6.778705,5.466985,3.376580,-7.591678,-0.637370,1.762110,2.525448,-5.172906,-6.031739,7.679017,-7.549739,-6.553065,-4.152373,1.612134,3.153117,7.420727,6.694486,-9.786919,-8.651833,8.107025,-9.445137,4.436399,-3.146608,-0.487898,3.026558,-0.741766,-7.078099,3.975702,8.009614,-5.145246,-3.093960,-6.340292,-6.193114,-7.973823,0.922604,1.630248,7.790689,1.698868,1.067365,-0.202287,8.700254,-1.406004,9.678391,9.204067,-3.061721,-1.893541,8.619567,6.968429,7.794740,5.036267,6.234160,-6.448908,7.435705,-9.280298,8.172644,2.319922,-5.682826,5.689402,7.657342,-0.758870,-0.343050,0.348741,5.649227,9.257799,-3.609125,-7.939549,-1.167676,-1.094392,-1.101594,-4.517564,-7.822508,4.415453,2.396969,-8.061312,5.611071,5.536260,0.418024,1.859989,8.993262,2.379386,6.349504,1.324435,-4.466930,-2.834678,5.691668,1.597846,-6.720755,-6.064059,7.762179,3.363616,2.315419,-1.425616,5.037660,3.746940,-7.511490,-9.982085,-4.637472,6.861217,-9.542077,-1.650697,-9.685471,9.185400,-0.468819,7.710675,-5.272451,-3.379097,1.854195,3.842536,-6.427618,1.521927,7.570644,7.066989,9.335417,3.080216,-4.469917,9.857880,6.329965,3.844810,-4.715235,7.445677,6.666808,-5.632165,-4.117844,2.139599,4.434757,9.514083,3.101167,-2.732801,-0.193025,-8.105619,6.121772,5.543695,-6.005053,-9.368226,0.138649,8.757685,4.434042,7.297228,7.804840,7.534668,3.548161,-5.859026,1.520611,-2.930295,-6.534990,-4.506702,-2.770056,-1.376846,-0.606333,-2.400077,-5.562569,5.615885,8.007615], dtype = "float32")#candidate|13700|(330,)|const|float32
call_13699 = relay.TupleGetItem(func_8589_call(relay.reshape(const_13700.astype('float32'), [15, 11, 2])), 0)
call_13701 = relay.TupleGetItem(func_8591_call(relay.reshape(const_13700.astype('float32'), [15, 11, 2])), 0)
uop_13704 = relay.sqrt(uop_13690.astype('float64')) # shape=(105, 9)
uop_13706 = relay.sqrt(uop_13692.astype('float64')) # shape=(105, 9)
output = relay.Tuple([call_13650,call_13680,bop_13683,call_13699,const_13700,uop_13704,])
output2 = relay.Tuple([call_13651,call_13681,bop_13686,call_13701,const_13700,uop_13706,])
func_13712 = relay.Function([var_13682,], output)
mod['func_13712'] = func_13712
mod = relay.transform.InferType()(mod)
var_13713 = relay.var("var_13713", dtype = "bool", shape = (105, 9))#candidate|13713|(105, 9)|var|bool
output = func_13712(var_13713)
func_13714 = relay.Function([var_13713], output)
mutated_mod['func_13714'] = func_13714
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8121_call = mod.get_global_var('func_8121')
func_8123_call = mutated_mod.get_global_var('func_8123')
call_13747 = func_8121_call()
call_13748 = func_8121_call()
output = relay.Tuple([call_13747,])
output2 = relay.Tuple([call_13748,])
func_13758 = relay.Function([], output)
mod['func_13758'] = func_13758
mod = relay.transform.InferType()(mod)
output = func_13758()
func_13759 = relay.Function([], output)
mutated_mod['func_13759'] = func_13759
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8789_call = mod.get_global_var('func_8789')
func_8791_call = mutated_mod.get_global_var('func_8791')
call_13797 = relay.TupleGetItem(func_8789_call(), 1)
call_13798 = relay.TupleGetItem(func_8791_call(), 1)
output = call_13797
output2 = call_13798
func_13803 = relay.Function([], output)
mod['func_13803'] = func_13803
mod = relay.transform.InferType()(mod)
mutated_mod['func_13803'] = func_13803
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13803_call = mutated_mod.get_global_var('func_13803')
call_13804 = func_13803_call()
output = call_13804
func_13805 = relay.Function([], output)
mutated_mod['func_13805'] = func_13805
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7136_call = mod.get_global_var('func_7136')
func_7137_call = mutated_mod.get_global_var('func_7137')
call_13843 = func_7136_call()
call_13844 = func_7136_call()
output = call_13843
output2 = call_13844
func_13845 = relay.Function([], output)
mod['func_13845'] = func_13845
mod = relay.transform.InferType()(mod)
output = func_13845()
func_13846 = relay.Function([], output)
mutated_mod['func_13846'] = func_13846
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10111_call = mod.get_global_var('func_10111')
func_10112_call = mutated_mod.get_global_var('func_10112')
call_13855 = func_10111_call()
call_13856 = func_10111_call()
func_5200_call = mod.get_global_var('func_5200')
func_5201_call = mutated_mod.get_global_var('func_5201')
call_13869 = relay.TupleGetItem(func_5200_call(), 0)
call_13870 = relay.TupleGetItem(func_5201_call(), 0)
uop_13877 = relay.cos(call_13855.astype('float64')) # shape=(14, 2, 10)
uop_13879 = relay.cos(call_13856.astype('float64')) # shape=(14, 2, 10)
output = relay.Tuple([call_13869,uop_13877,])
output2 = relay.Tuple([call_13870,uop_13879,])
func_13889 = relay.Function([], output)
mod['func_13889'] = func_13889
mod = relay.transform.InferType()(mod)
output = func_13889()
func_13890 = relay.Function([], output)
mutated_mod['func_13890'] = func_13890
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12751_call = mod.get_global_var('func_12751')
func_12753_call = mutated_mod.get_global_var('func_12753')
call_13901 = func_12751_call()
call_13902 = func_12751_call()
output = relay.Tuple([call_13901,])
output2 = relay.Tuple([call_13902,])
func_13920 = relay.Function([], output)
mod['func_13920'] = func_13920
mod = relay.transform.InferType()(mod)
output = func_13920()
func_13921 = relay.Function([], output)
mutated_mod['func_13921'] = func_13921
mutated_mod = relay.transform.InferType()(mutated_mod)
func_9943_call = mod.get_global_var('func_9943')
func_9945_call = mutated_mod.get_global_var('func_9945')
call_13968 = relay.TupleGetItem(func_9943_call(), 0)
call_13969 = relay.TupleGetItem(func_9945_call(), 0)
output = call_13968
output2 = call_13969
func_13987 = relay.Function([], output)
mod['func_13987'] = func_13987
mod = relay.transform.InferType()(mod)
output = func_13987()
func_13988 = relay.Function([], output)
mutated_mod['func_13988'] = func_13988
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11742_call = mod.get_global_var('func_11742')
func_11744_call = mutated_mod.get_global_var('func_11744')
call_14052 = relay.TupleGetItem(func_11742_call(), 3)
call_14053 = relay.TupleGetItem(func_11744_call(), 3)
output = relay.Tuple([call_14052,])
output2 = relay.Tuple([call_14053,])
func_14063 = relay.Function([], output)
mod['func_14063'] = func_14063
mod = relay.transform.InferType()(mod)
output = func_14063()
func_14064 = relay.Function([], output)
mutated_mod['func_14064'] = func_14064
mutated_mod = relay.transform.InferType()(mutated_mod)
const_14088 = relay.const([[[-8.283737,3.767991,3.154222,-7.873838],[1.950724,-3.959986,3.280588,-0.468207]],[[3.231561,3.700380,0.550780,-1.659396],[2.881616,2.152002,6.507558,-3.831724]],[[2.183412,4.003126,-7.160793,2.599853],[5.649493,-7.265038,4.667493,7.503501]],[[9.986056,-0.380165,-6.128210,5.163082],[-6.187327,4.530205,3.636399,6.694026]],[[0.238473,3.654487,-0.862868,6.315829],[-7.618103,5.351083,-9.422561,-1.684706]],[[8.874492,-7.543149,8.230731,3.219990],[3.191241,7.137057,2.587789,7.462566]],[[-3.763394,-0.951899,6.648442,-5.190529],[-1.748983,4.463250,-9.142570,-3.626723]],[[0.925351,-0.892358,-3.552145,-4.234203],[-5.765085,-8.475990,-8.167775,4.635794]],[[-5.953270,-7.517066,4.179651,-7.844187],[-4.353807,1.059278,3.456911,-8.341981]],[[-6.153533,-4.417999,-6.369921,-0.118306],[-0.013424,-4.377037,6.778986,0.758789]],[[-7.531455,0.795091,-6.837511,0.090376],[-5.535314,-4.903546,-3.188419,-6.166143]]], dtype = "float32")#candidate|14088|(11, 2, 4)|const|float32
uop_14089 = relay.rsqrt(const_14088.astype('float32')) # shape=(11, 2, 4)
output = uop_14089
output2 = uop_14089
func_14095 = relay.Function([], output)
mod['func_14095'] = func_14095
mod = relay.transform.InferType()(mod)
mutated_mod['func_14095'] = func_14095
mutated_mod = relay.transform.InferType()(mutated_mod)
func_14095_call = mutated_mod.get_global_var('func_14095')
call_14096 = func_14095_call()
output = call_14096
func_14097 = relay.Function([], output)
mutated_mod['func_14097'] = func_14097
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11447_call = mod.get_global_var('func_11447')
func_11449_call = mutated_mod.get_global_var('func_11449')
call_14106 = func_11447_call()
call_14107 = func_11447_call()
func_1497_call = mod.get_global_var('func_1497')
func_1499_call = mutated_mod.get_global_var('func_1499')
var_14113 = relay.var("var_14113", dtype = "float64", shape = (520, 1))#candidate|14113|(520, 1)|var|float64
call_14112 = relay.TupleGetItem(func_1497_call(relay.reshape(var_14113.astype('float64'), [8, 5, 13])), 3)
call_14114 = relay.TupleGetItem(func_1499_call(relay.reshape(var_14113.astype('float64'), [8, 5, 13])), 3)
bop_14122 = relay.left_shift(var_14113.astype('int16'), call_14112.astype('int16')) # shape=(520, 1280)
bop_14125 = relay.left_shift(var_14113.astype('int16'), call_14114.astype('int16')) # shape=(520, 1280)
output = relay.Tuple([call_14106,bop_14122,])
output2 = relay.Tuple([call_14107,bop_14125,])
func_14133 = relay.Function([var_14113,], output)
mod['func_14133'] = func_14133
mod = relay.transform.InferType()(mod)
var_14134 = relay.var("var_14134", dtype = "float64", shape = (520, 1))#candidate|14134|(520, 1)|var|float64
output = func_14133(var_14134)
func_14135 = relay.Function([var_14134], output)
mutated_mod['func_14135'] = func_14135
mutated_mod = relay.transform.InferType()(mutated_mod)
var_14145 = relay.var("var_14145", dtype = "int8", shape = (10, 13, 15))#candidate|14145|(10, 13, 15)|var|int8
var_14146 = relay.var("var_14146", dtype = "int8", shape = (10, 13, 15))#candidate|14146|(10, 13, 15)|var|int8
bop_14147 = relay.greater(var_14145.astype('bool'), relay.reshape(var_14146.astype('bool'), relay.shape_of(var_14145))) # shape=(10, 13, 15)
func_10703_call = mod.get_global_var('func_10703')
func_10705_call = mutated_mod.get_global_var('func_10705')
call_14151 = relay.TupleGetItem(func_10703_call(), 0)
call_14152 = relay.TupleGetItem(func_10705_call(), 0)
output = relay.Tuple([bop_14147,call_14151,])
output2 = relay.Tuple([bop_14147,call_14152,])
func_14158 = relay.Function([var_14145,var_14146,], output)
mod['func_14158'] = func_14158
mod = relay.transform.InferType()(mod)
mutated_mod['func_14158'] = func_14158
mutated_mod = relay.transform.InferType()(mutated_mod)
func_14158_call = mutated_mod.get_global_var('func_14158')
var_14160 = relay.var("var_14160", dtype = "int8", shape = (10, 13, 15))#candidate|14160|(10, 13, 15)|var|int8
var_14161 = relay.var("var_14161", dtype = "int8", shape = (10, 13, 15))#candidate|14161|(10, 13, 15)|var|int8
call_14159 = func_14158_call(var_14160,var_14161,)
output = call_14159
func_14162 = relay.Function([var_14160,var_14161,], output)
mutated_mod['func_14162'] = func_14162
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8623_call = mod.get_global_var('func_8623')
func_8625_call = mutated_mod.get_global_var('func_8625')
call_14180 = relay.TupleGetItem(func_8623_call(), 1)
call_14181 = relay.TupleGetItem(func_8625_call(), 1)
func_6794_call = mod.get_global_var('func_6794')
func_6795_call = mutated_mod.get_global_var('func_6795')
call_14194 = relay.TupleGetItem(func_6794_call(), 0)
call_14195 = relay.TupleGetItem(func_6795_call(), 0)
output = relay.Tuple([call_14180,call_14194,])
output2 = relay.Tuple([call_14181,call_14195,])
func_14199 = relay.Function([], output)
mod['func_14199'] = func_14199
mod = relay.transform.InferType()(mod)
output = func_14199()
func_14200 = relay.Function([], output)
mutated_mod['func_14200'] = func_14200
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13417_call = mod.get_global_var('func_13417')
func_13418_call = mutated_mod.get_global_var('func_13418')
call_14201 = relay.TupleGetItem(func_13417_call(), 0)
call_14202 = relay.TupleGetItem(func_13418_call(), 0)
output = relay.Tuple([call_14201,])
output2 = relay.Tuple([call_14202,])
func_14223 = relay.Function([], output)
mod['func_14223'] = func_14223
mod = relay.transform.InferType()(mod)
mutated_mod['func_14223'] = func_14223
mutated_mod = relay.transform.InferType()(mutated_mod)
func_14223_call = mutated_mod.get_global_var('func_14223')
call_14224 = func_14223_call()
output = call_14224
func_14225 = relay.Function([], output)
mutated_mod['func_14225'] = func_14225
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11491_call = mod.get_global_var('func_11491')
func_11493_call = mutated_mod.get_global_var('func_11493')
call_14234 = func_11491_call()
call_14235 = func_11491_call()
func_5178_call = mod.get_global_var('func_5178')
func_5179_call = mutated_mod.get_global_var('func_5179')
call_14246 = relay.TupleGetItem(func_5178_call(), 0)
call_14247 = relay.TupleGetItem(func_5179_call(), 0)
func_8815_call = mod.get_global_var('func_8815')
func_8816_call = mutated_mod.get_global_var('func_8816')
call_14258 = func_8815_call()
call_14259 = func_8815_call()
output = relay.Tuple([call_14234,call_14246,call_14258,])
output2 = relay.Tuple([call_14235,call_14247,call_14259,])
func_14265 = relay.Function([], output)
mod['func_14265'] = func_14265
mod = relay.transform.InferType()(mod)
output = func_14265()
func_14266 = relay.Function([], output)
mutated_mod['func_14266'] = func_14266
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13569_call = mod.get_global_var('func_13569')
func_13571_call = mutated_mod.get_global_var('func_13571')
call_14302 = relay.TupleGetItem(func_13569_call(), 1)
call_14303 = relay.TupleGetItem(func_13571_call(), 1)
func_12067_call = mod.get_global_var('func_12067')
func_12069_call = mutated_mod.get_global_var('func_12069')
call_14306 = relay.TupleGetItem(func_12067_call(), 0)
call_14307 = relay.TupleGetItem(func_12069_call(), 0)
func_8253_call = mod.get_global_var('func_8253')
func_8255_call = mutated_mod.get_global_var('func_8255')
call_14328 = func_8253_call()
call_14329 = func_8253_call()
output = relay.Tuple([call_14302,call_14306,call_14328,])
output2 = relay.Tuple([call_14303,call_14307,call_14329,])
func_14339 = relay.Function([], output)
mod['func_14339'] = func_14339
mod = relay.transform.InferType()(mod)
output = func_14339()
func_14340 = relay.Function([], output)
mutated_mod['func_14340'] = func_14340
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6517_call = mod.get_global_var('func_6517')
func_6519_call = mutated_mod.get_global_var('func_6519')
call_14341 = relay.TupleGetItem(func_6517_call(), 0)
call_14342 = relay.TupleGetItem(func_6519_call(), 0)
func_5000_call = mod.get_global_var('func_5000')
func_5005_call = mutated_mod.get_global_var('func_5005')
const_14388 = relay.const([-7.455494,-3.484547,5.533726,3.545027,-5.989219,1.914191,7.474228,-6.379653,4.555771,0.518374,0.367975,0.515750,3.871017,1.759703,0.133398,-4.606797,-3.700424,8.772295,-6.596238,8.243484,-5.309708,7.852413,-3.135520,-2.468228,6.648540,5.723023,-5.038131,-7.733471,8.753114,-3.174974,7.118016,-7.064738,7.750865,2.725890,-5.315104,-6.246567,-4.908991,6.668307,6.082167,-7.540621,1.657166,1.064192,9.764841,4.692169,-3.481230,6.468697,-6.085856,5.218895,8.843148,-0.542589,9.649949,6.481019,-3.748228,5.946415,-0.065172,-8.650932,1.617923,4.444517,7.898171,-6.266413,-1.936773,8.765227,3.478751,-1.565471,-5.406552,1.777412,-3.239548,7.745609,5.604656,-6.853892,-2.013378,1.301785,9.978708,-2.497818,5.089681,-7.773778,7.276904,-5.858900,-7.241630,0.253276,4.567776,-2.660567,5.583352,-6.602163,-6.084365,-2.418176,4.336615,5.307447,-0.053654,9.267819,-5.973444,9.426503,-4.031336,-6.000940,1.211161,7.277041,-4.945913,-6.494017,1.331556,-4.716332,9.522324,-3.108048,-0.205122,8.773848,-0.859192,-0.697871,-1.049660,-9.566786,-1.080314,2.201502,-1.264055,-2.061543,8.087680,4.741162,-9.417682,-2.613558,-7.982227,-5.144525,-9.631571,-8.112044,2.280055,3.380209,-6.365952,-5.239301,5.623914,-6.512739,-8.087878,-9.993063,0.550506,-0.784311,-2.297005,-7.506972,1.085329,4.289026,-2.218865,-7.485520,3.239216,-4.124647,0.926631,-8.261010,-0.065887,5.926761,6.749868,-1.143174,0.750791,-4.778585,-6.951292,-5.130487,-9.800183,9.763235,3.894562,-6.852546,-5.647864,2.148774,9.998127,-6.430311,-2.469953,-9.206873,1.615022,6.270078,5.080241,-8.061031,-9.164554,2.069336,9.321803,4.189249,8.116662,2.555187,-7.129419,7.770809,-2.353888,-8.635372,1.015424,-6.406596,3.608323,3.739994,-1.463539,-7.378031,7.791338,0.927194,6.635770,5.513863,-1.256245,-1.172189,8.890053,-1.415394,3.083218,3.393434,5.735586,5.501007,1.160645,2.630928,4.171202,-4.606629,4.351051,7.046974,-4.309965,6.330130,8.600191,-7.968403,-1.331224,-2.907198,4.843685,-0.515095,9.485634,7.610818,6.706994,-5.697000,4.385139,3.489584,-1.049416,8.069135,1.614832,6.826995,-7.256184,-2.892381,-6.347781,-3.728431,-7.916823,-2.918034,-2.185433,-9.284897,-5.219169,-7.545571,1.071408,6.584717,-3.281945,-0.304769,6.176192,-7.988807,1.825251,-5.953197,5.031580,3.866944,-2.468026,-9.919063,-6.817601,-1.381098,0.220346,-9.958334,-4.906297,9.200172,-0.709182,-6.171909,4.801992,1.623970,5.138654,5.881258,-5.960790,-9.607636,-4.587112,7.373648,9.413265,-8.213346,6.433628,5.640645,-1.627299,-1.917742,2.971203,-2.023121,-6.479646,6.446620,-3.657282,-2.306298,-8.040579,1.597910,5.146789,7.250836,-9.379893,-1.068359,-1.268477,8.673920,-8.646734,7.703709,-3.260602,-2.223928,7.589527,4.976247,-0.682694,-8.352164,-7.204956,-6.966014,-3.675712,2.126975,8.365174,-2.522470,7.266074,-4.033137,-1.564214,0.289009,1.615301,6.804678,-4.329117,-2.597415,-7.234585,1.843306,-8.253536,4.895379,2.219016,0.774696,-4.452326,-1.828040,4.507954,6.479698,-6.678365,4.597997,-9.728087,-6.349383,1.569692,-4.194162,6.749206,-7.672622,4.213803,-4.247654,-0.524049,5.071699,-1.113249,9.793675,-2.349772,2.025403,5.634972,-7.407378,7.847248,-2.497501,-7.354508,-3.448067,-7.371067,8.000341,-9.743809,-8.690938,-8.319497,9.607081,8.697381,-6.627202,9.608640,-6.699705,-6.381659,8.277805,3.785390,-7.166874,-0.440898,0.816400,-3.788462,-9.980589,6.030625,1.006597,-5.451477,-2.375795,2.841080,6.434769,-5.119570,-0.735868,-1.168527,-5.038260,3.510858,5.650612,9.072698,2.335049,-5.235083,9.618197,-7.896105,2.371893,5.241950,8.652134,-7.497607,-8.338193,-3.567585,-6.655319,-0.632538,4.500483,3.424942,3.426536,1.670907,-4.462725,-1.426583,1.994543,4.640029,9.282012,-7.987722,7.699476,5.865205,3.725176,-6.615585,-1.536267,-7.550670,6.149620,5.140738,5.737661,0.519049,-5.518118,3.751441,-5.027642,-8.241743,6.335489,-6.349873,2.141196,-6.555370,-0.779586,-1.732803,4.213734,-0.165602,4.527488,9.519796,-4.417336,-3.562154,-1.644563,8.250462,-6.063031,-7.071122,-9.128619,3.176259,-9.593930,7.880115,-5.333730,-3.000997,-6.663707,-6.725840,-0.306555,-5.624153,-2.658351,0.316618,-5.602265,-9.276840,-2.848581,0.812315,9.396693,-7.815308,-3.472793,-9.053471,-6.912358,4.900742,3.721725,-5.243776,0.517528,2.449951,4.097263,6.269307,2.173118,8.061079,-2.021513,-8.975964,2.044285,-5.751953,-6.464816,-4.955870,6.049485,-8.419748,-4.612501,-2.457454,3.803120,6.766519,-2.423203,5.037468,-3.682222,0.923616,0.150598,8.973770,-0.018740,-4.306469,-3.330126,5.875381,-0.600765,5.536505,-5.920681,4.986540,1.432900,9.351021,7.425618,8.152283,9.117304,-1.906552,-1.811587,1.857949,4.606577,-2.620424,-1.576515,6.842420,8.375509,8.888137,-4.688390,3.537263,-3.770688,9.396906,0.663843,0.313295,5.138722,3.165011,1.582299,3.994394,7.792700,1.137664,6.368308,-8.215228,4.980941,5.330611,-0.144288,8.554283,7.226056,-5.621061,5.801211,8.561507,-8.224004,-5.232410,5.369543,-3.232297,5.130831,-3.332591,5.542397,8.207037,1.207434,-0.078067,-3.503718,-9.257368,8.709306,-9.467557,-2.199918,7.546117,7.863767,2.295905,6.040273,7.357866,2.236165,-1.133540,-2.612471,6.176176,-9.651719,-6.312705,1.898252,-6.892857,8.460837,4.450619,-5.625102,-2.981566,-4.628330,-4.846582,-2.256056,3.044055,1.162491,-0.063158,8.986745,7.300857,0.646781,-1.895488,-2.289825,5.685823,-9.087349,-5.305157,-2.059650,-1.166379,-1.784813,-9.114216,-0.071077,-3.568465,8.258367,7.542058,-0.595135,-0.415824,-8.083116,-6.535256,6.905747,8.010354,-3.950439,-5.160370,-7.023537,0.009166,5.871788,8.461320,2.498155,-4.247723,-9.817396,-7.438787,-0.078519,3.278288,2.297065,-5.666013,3.211244,-8.826257,3.861743,-3.624271,6.293233,-1.570336,-2.064044,-6.692492,0.285398,-8.163893,3.344959,-4.170702,-0.178437,-1.715629,1.460817,-9.329929,-4.724855,-5.197026,-4.930421,-4.192612,-5.773407,-3.696106,-3.747945,0.150417,0.287259,-4.813964,-3.482085,8.245471,-4.093397,-3.026166,-4.667753,-1.415885,-7.031162,-4.629442,-6.169712,-8.920693,9.536185,6.919528,-2.898104,5.704376,-6.470051,-1.609152,-9.488200,0.864386,3.884701,-0.420676,3.524799,7.789582,7.101826,-9.065946,-3.559692,-3.877181,4.080536,7.275330,-9.551917,-8.498621,-1.275260,8.058957,-8.988747,3.947027,5.551344,8.721572,1.063858,-5.630816,-9.181029,-7.416098,9.735096,-1.725778,-0.946705,-9.435500,-4.894406,9.815436,5.770190,5.201055,-3.287356,-1.483524,-4.640182,-3.977879,-8.194769,-2.023578,2.000771,0.625859,0.826783,5.481167,7.497900,7.889743,-3.787622,7.509227,6.814619,5.584791,7.630501,6.781899,5.013756,9.279776,-8.919424,-2.844446,-7.023757,5.911766,7.122616,9.952704,-9.809737,-4.926010,-1.132936,5.142629,-4.620248,8.930026,-9.871795,9.112372,-6.168356,2.751317,-0.661552,-1.899503,0.571917,8.318564,-2.513731,9.616722,7.517873,-3.905422,2.464273,8.464211,-1.623582,-1.866322,-0.017672,7.902280,-1.539087,1.144902,-3.469404,-4.058302,8.638521,9.455554,-2.801239,9.704480,-4.618466,5.477432,7.572904,-7.426136,6.757477,-5.659317,-0.853534,-5.506869,-3.606657,-0.644753,1.541541,-0.339852,-0.120118,7.117904,-8.522642,-4.693545,-8.390465,5.415959,-8.340132,2.639331,-4.605899,0.159854,1.160496,2.023731,-4.728944,9.927131,-2.901836,-2.131489,0.211817,-7.739703,-4.679388,-3.501933,-9.283650,-0.023662,6.484666,5.049575,9.624851,6.458704,-7.746101,-4.532576,-9.025619,8.813637,4.333866,7.895051,2.067003,4.529347,-0.478528,-6.989677,7.003892,2.875556,-8.760309,-3.801166,-5.058789,8.275435,-3.427444,9.842796,2.660812,-2.143521,-4.612766,0.793159,8.091528,-5.519621,-4.715396,-7.675477,4.594466,3.973766,-6.920734,-0.900796,6.643584,-5.407116,6.995166,-4.876505,-6.770299,-4.894548,-0.515672,-2.169300,-5.543711,0.560817,7.014571,-0.221474,9.131765,-4.073349,4.980685,-0.707747,-0.412982], dtype = "float32")#candidate|14388|(792,)|const|float32
var_14389 = relay.var("var_14389", dtype = "uint16", shape = (336,))#candidate|14389|(336,)|var|uint16
var_14390 = relay.var("var_14390", dtype = "int8", shape = (729,))#candidate|14390|(729,)|var|int8
call_14387 = relay.TupleGetItem(func_5000_call(relay.reshape(const_14388.astype('float32'), [792,]), relay.reshape(var_14389.astype('uint16'), [336,]), relay.reshape(var_14390.astype('int8'), [729,]), ), 1)
call_14391 = relay.TupleGetItem(func_5005_call(relay.reshape(const_14388.astype('float32'), [792,]), relay.reshape(var_14389.astype('uint16'), [336,]), relay.reshape(var_14390.astype('int8'), [729,]), ), 1)
func_11259_call = mod.get_global_var('func_11259')
func_11261_call = mutated_mod.get_global_var('func_11261')
call_14398 = relay.TupleGetItem(func_11259_call(), 2)
call_14399 = relay.TupleGetItem(func_11261_call(), 2)
func_10111_call = mod.get_global_var('func_10111')
func_10112_call = mutated_mod.get_global_var('func_10112')
call_14407 = func_10111_call()
call_14408 = func_10111_call()
uop_14417 = relay.sin(var_14390.astype('float32')) # shape=(729,)
uop_14419 = relay.sigmoid(uop_14417.astype('float32')) # shape=(729,)
output = relay.Tuple([call_14341,call_14387,const_14388,var_14389,call_14398,call_14407,uop_14419,])
output2 = relay.Tuple([call_14342,call_14391,const_14388,var_14389,call_14399,call_14408,uop_14419,])
func_14435 = relay.Function([var_14389,var_14390,], output)
mod['func_14435'] = func_14435
mod = relay.transform.InferType()(mod)
var_14436 = relay.var("var_14436", dtype = "uint16", shape = (336,))#candidate|14436|(336,)|var|uint16
var_14437 = relay.var("var_14437", dtype = "int8", shape = (729,))#candidate|14437|(729,)|var|int8
output = func_14435(var_14436,var_14437,)
func_14438 = relay.Function([var_14436,var_14437,], output)
mutated_mod['func_14438'] = func_14438
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5642_call = mod.get_global_var('func_5642')
func_5644_call = mutated_mod.get_global_var('func_5644')
call_14466 = relay.TupleGetItem(func_5642_call(), 2)
call_14467 = relay.TupleGetItem(func_5644_call(), 2)
output = call_14466
output2 = call_14467
func_14472 = relay.Function([], output)
mod['func_14472'] = func_14472
mod = relay.transform.InferType()(mod)
output = func_14472()
func_14473 = relay.Function([], output)
mutated_mod['func_14473'] = func_14473
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8388_call = mod.get_global_var('func_8388')
func_8390_call = mutated_mod.get_global_var('func_8390')
call_14502 = relay.TupleGetItem(func_8388_call(), 0)
call_14503 = relay.TupleGetItem(func_8390_call(), 0)
output = relay.Tuple([call_14502,])
output2 = relay.Tuple([call_14503,])
func_14526 = relay.Function([], output)
mod['func_14526'] = func_14526
mod = relay.transform.InferType()(mod)
mutated_mod['func_14526'] = func_14526
mutated_mod = relay.transform.InferType()(mutated_mod)
func_14526_call = mutated_mod.get_global_var('func_14526')
call_14527 = func_14526_call()
output = call_14527
func_14528 = relay.Function([], output)
mutated_mod['func_14528'] = func_14528
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13062_call = mod.get_global_var('func_13062')
func_13063_call = mutated_mod.get_global_var('func_13063')
call_14556 = func_13062_call()
call_14557 = func_13062_call()
output = relay.Tuple([call_14556,])
output2 = relay.Tuple([call_14557,])
func_14568 = relay.Function([], output)
mod['func_14568'] = func_14568
mod = relay.transform.InferType()(mod)
output = func_14568()
func_14569 = relay.Function([], output)
mutated_mod['func_14569'] = func_14569
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10070_call = mod.get_global_var('func_10070')
func_10072_call = mutated_mod.get_global_var('func_10072')
call_14575 = relay.TupleGetItem(func_10070_call(), 1)
call_14576 = relay.TupleGetItem(func_10072_call(), 1)
func_12507_call = mod.get_global_var('func_12507')
func_12508_call = mutated_mod.get_global_var('func_12508')
call_14580 = func_12507_call()
call_14581 = func_12507_call()
func_12778_call = mod.get_global_var('func_12778')
func_12779_call = mutated_mod.get_global_var('func_12779')
call_14587 = func_12778_call()
call_14588 = func_12778_call()
output = relay.Tuple([call_14575,call_14580,call_14587,])
output2 = relay.Tuple([call_14576,call_14581,call_14588,])
func_14621 = relay.Function([], output)
mod['func_14621'] = func_14621
mod = relay.transform.InferType()(mod)
mutated_mod['func_14621'] = func_14621
mutated_mod = relay.transform.InferType()(mutated_mod)
func_14621_call = mutated_mod.get_global_var('func_14621')
call_14622 = func_14621_call()
output = call_14622
func_14623 = relay.Function([], output)
mutated_mod['func_14623'] = func_14623
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8866_call = mod.get_global_var('func_8866')
func_8868_call = mutated_mod.get_global_var('func_8868')
call_14705 = relay.TupleGetItem(func_8866_call(), 1)
call_14706 = relay.TupleGetItem(func_8868_call(), 1)
output = call_14705
output2 = call_14706
func_14707 = relay.Function([], output)
mod['func_14707'] = func_14707
mod = relay.transform.InferType()(mod)
output = func_14707()
func_14708 = relay.Function([], output)
mutated_mod['func_14708'] = func_14708
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8253_call = mod.get_global_var('func_8253')
func_8255_call = mutated_mod.get_global_var('func_8255')
call_14722 = func_8253_call()
call_14723 = func_8253_call()
var_14727 = relay.var("var_14727", dtype = "float64", shape = (64,))#candidate|14727|(64,)|var|float64
bop_14728 = relay.minimum(call_14722.astype('float32'), relay.reshape(var_14727.astype('float32'), relay.shape_of(call_14722))) # shape=(64,)
bop_14731 = relay.minimum(call_14723.astype('float32'), relay.reshape(var_14727.astype('float32'), relay.shape_of(call_14723))) # shape=(64,)
func_12843_call = mod.get_global_var('func_12843')
func_12844_call = mutated_mod.get_global_var('func_12844')
call_14732 = relay.TupleGetItem(func_12843_call(), 0)
call_14733 = relay.TupleGetItem(func_12844_call(), 0)
func_11199_call = mod.get_global_var('func_11199')
func_11200_call = mutated_mod.get_global_var('func_11200')
call_14746 = func_11199_call()
call_14747 = func_11199_call()
func_8235_call = mod.get_global_var('func_8235')
func_8237_call = mutated_mod.get_global_var('func_8237')
call_14777 = func_8235_call()
call_14778 = func_8235_call()
func_7656_call = mod.get_global_var('func_7656')
func_7657_call = mutated_mod.get_global_var('func_7657')
call_14783 = relay.TupleGetItem(func_7656_call(), 1)
call_14784 = relay.TupleGetItem(func_7657_call(), 1)
func_6666_call = mod.get_global_var('func_6666')
func_6668_call = mutated_mod.get_global_var('func_6668')
call_14817 = relay.TupleGetItem(func_6666_call(), 0)
call_14818 = relay.TupleGetItem(func_6668_call(), 0)
func_13845_call = mod.get_global_var('func_13845')
func_13846_call = mutated_mod.get_global_var('func_13846')
call_14820 = func_13845_call()
call_14821 = func_13845_call()
func_4770_call = mod.get_global_var('func_4770')
func_4772_call = mutated_mod.get_global_var('func_4772')
call_14822 = relay.TupleGetItem(func_4770_call(), 0)
call_14823 = relay.TupleGetItem(func_4772_call(), 0)
func_4770_call = mod.get_global_var('func_4770')
func_4772_call = mutated_mod.get_global_var('func_4772')
call_14840 = relay.TupleGetItem(func_4770_call(), 2)
call_14841 = relay.TupleGetItem(func_4772_call(), 2)
output = relay.Tuple([bop_14728,call_14732,call_14746,call_14777,call_14783,call_14817,call_14820,call_14822,call_14840,])
output2 = relay.Tuple([bop_14731,call_14733,call_14747,call_14778,call_14784,call_14818,call_14821,call_14823,call_14841,])
func_14847 = relay.Function([var_14727,], output)
mod['func_14847'] = func_14847
mod = relay.transform.InferType()(mod)
mutated_mod['func_14847'] = func_14847
mutated_mod = relay.transform.InferType()(mutated_mod)
var_14848 = relay.var("var_14848", dtype = "float64", shape = (64,))#candidate|14848|(64,)|var|float64
func_14847_call = mutated_mod.get_global_var('func_14847')
call_14849 = func_14847_call(var_14848)
output = call_14849
func_14850 = relay.Function([var_14848], output)
mutated_mod['func_14850'] = func_14850
mutated_mod = relay.transform.InferType()(mutated_mod)
const_14868 = relay.const([[[-3.948515,-6.422483,3.582349,-0.895777,-8.632223],[3.745858,6.305841,-0.164169,6.645481,-0.070071],[5.442647,-8.235366,6.642505,6.210161,4.346181],[3.443865,0.479181,-7.137955,6.958199,9.609689],[-8.255973,-7.921849,-4.426118,7.566844,-9.045042],[-3.420640,9.705191,7.560873,7.228621,8.111865]],[[-4.607925,5.542093,-2.312066,-1.069741,-3.230945],[6.178708,2.062349,-2.458910,-1.660047,0.266651],[3.782944,-7.225219,4.384690,-8.130345,-1.734405],[7.549700,-5.644104,-7.776459,-0.568130,5.793229],[2.429390,-3.200368,-4.848833,-5.995529,-0.196006],[5.010960,-0.374377,-5.434176,-3.936510,-2.774523]],[[-7.038395,8.387075,-5.267718,9.241381,-7.379045],[0.959227,1.468545,-7.276874,8.982441,-0.784103],[3.403487,2.483139,8.848364,8.559299,-9.717526],[-3.722237,4.949246,-4.892135,-1.369570,4.014223],[-9.143068,-5.562273,-6.937441,7.452617,7.417984],[2.149801,-2.423216,-6.024688,0.714111,8.693340]],[[-0.760748,-5.448782,3.874750,1.892963,4.078599],[-9.627053,-4.926289,0.881267,-4.175212,-1.515532],[7.353444,-4.092080,-6.750602,9.558575,-2.225423],[-3.630706,0.042458,-0.367479,1.309043,-0.259469],[-4.064146,-8.200359,6.531285,1.756402,-9.875735],[5.959623,-8.814619,4.493249,1.576453,-3.357992]]], dtype = "float64")#candidate|14868|(4, 6, 5)|const|float64
uop_14869 = relay.cosh(const_14868.astype('float64')) # shape=(4, 6, 5)
func_11055_call = mod.get_global_var('func_11055')
func_11057_call = mutated_mod.get_global_var('func_11057')
call_14874 = relay.TupleGetItem(func_11055_call(), 5)
call_14875 = relay.TupleGetItem(func_11057_call(), 5)
output = relay.Tuple([uop_14869,call_14874,])
output2 = relay.Tuple([uop_14869,call_14875,])
func_14885 = relay.Function([], output)
mod['func_14885'] = func_14885
mod = relay.transform.InferType()(mod)
output = func_14885()
func_14886 = relay.Function([], output)
mutated_mod['func_14886'] = func_14886
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6666_call = mod.get_global_var('func_6666')
func_6668_call = mutated_mod.get_global_var('func_6668')
call_14912 = relay.TupleGetItem(func_6666_call(), 0)
call_14913 = relay.TupleGetItem(func_6668_call(), 0)
output = relay.Tuple([call_14912,])
output2 = relay.Tuple([call_14913,])
func_14917 = relay.Function([], output)
mod['func_14917'] = func_14917
mod = relay.transform.InferType()(mod)
mutated_mod['func_14917'] = func_14917
mutated_mod = relay.transform.InferType()(mutated_mod)
func_14917_call = mutated_mod.get_global_var('func_14917')
call_14918 = func_14917_call()
output = call_14918
func_14919 = relay.Function([], output)
mutated_mod['func_14919'] = func_14919
mutated_mod = relay.transform.InferType()(mutated_mod)
func_14707_call = mod.get_global_var('func_14707')
func_14708_call = mutated_mod.get_global_var('func_14708')
call_14945 = func_14707_call()
call_14946 = func_14707_call()
output = call_14945
output2 = call_14946
func_14997 = relay.Function([], output)
mod['func_14997'] = func_14997
mod = relay.transform.InferType()(mod)
output = func_14997()
func_14998 = relay.Function([], output)
mutated_mod['func_14998'] = func_14998
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13054_call = mod.get_global_var('func_13054')
func_13056_call = mutated_mod.get_global_var('func_13056')
call_15001 = func_13054_call()
call_15002 = func_13054_call()
func_13712_call = mod.get_global_var('func_13712')
func_13714_call = mutated_mod.get_global_var('func_13714')
const_15004 = relay.const([False,True,False,False,False,False,True,False,True,True,False,False,False,True,False,True,False,False,False,True,True,False,True,True,False,True,True,True,True,True,True,True,False,False,True,False,False,False,True,False,True,True,False,False,False,False,True,False,True,True,False,False,True,True,True,True,False,False,False,True,True,True,True,True,True,False,False,False,True,True,False,False,False,False,False,False,True,True,False,False,False,False,False,True,False,True,False,False,False,False,False,True,False,True,True,True,True,True,True,False,False,True,True,False,False,True,False,True,True,False,True,True,True,True,True,True,False,True,True,False,False,True,False,False,False,True,True,False,False,True,False,False,False,True,False,False,True,False,True,False,False,False,True,True,True,False,False,False,False,False,False,False,True,False,False,False,True,True,False,True,False,False,True,True,False,True,True,True,True,True,True,False,True,False,True,True,True,False,True,True,False,True,True,True,False,True,True,True,False,True,True,True,False,False,False,False,False,True,True,True,False,False,True,False,False,True,True,True,True,False,False,True,True,True,False,True,False,False,True,True,True,False,False,True,False,False,True,False,True,False,False,False,False,True,False,True,False,True,False,True,True,True,False,False,False,True,False,True,True,True,False,False,True,False,False,True,False,True,False,True,False,False,True,False,True,True,True,False,False,False,True,True,True,False,False,True,True,False,False,False,True,False,False,False,False,False,True,False,False,True,False,False,True,True,False,True,True,False,True,True,False,False,True,True,False,True,False,True,True,True,True,False,True,False,False,False,False,True,False,False,True,False,False,False,False,True,True,True,True,False,False,True,True,True,False,True,False,False,True,True,True,False,True,True,False,True,True,True,True,True,True,False,False,False,False,False,True,False,True,False,False,True,True,True,True,True,True,True,True,False,True,True,False,False,False,True,False,True,False,True,False,True,True,False,True,True,False,False,True,True,False,False,True,True,True,False,False,False,True,False,False,False,True,True,True,True,False,True,False,False,False,False,True,True,True,True,False,True,True,False,True,False,False,False,True,True,False,False,False,True,False,False,False,True,True,True,False,True,False,False,True,False,False,True,True,True,False,True,True,True,True,False,True,True,False,True,True,False,True,True,True,True,True,True,False,True,False,False,False,False,False,False,False,False,True,True,False,True,True,True,False,False,True,False,False,True,True,True,True,False,False,False,True,True,False,True,False,False,True,True,True,True,True,True,True,False,True,True,True,False,False,True,False,True,True,False,False,False,True,True,False,True,False,True,True,False,False,True,False,True,False,True,False,True,True,False,True,False,True,False,False,False,False,False,True,True,True,False,False,False,True,False,False,True,False,True,True,False,False,False,False,True,True,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False,True,False,False,True,False,False,False,False,True,False,True,False,False,True,True,True,True,True,False,False,True,False,False,False,True,False,False,False,True,False,False,True,False,False,False,True,True,False,True,False,False,False,False,False,True,True,False,True,True,False,True,False,False,True,False,True,True,True,True,False,True,True,True,False,False,True,True,True,False,False,True,False,True,False,True,False,True,True,True,False,True,False,True,True,True,True,True,False,False,False,True,False,False,False,False,False,False,True,True,False,True,False,False,False,False,True,True,False,False,False,False,True,True,True,True,False,True,False,True,True,False,False,True,False,False,True,False,False,False,True,False,False,False,True,False,False,False,True,False,True,False,True,False,True,False,True,True,True,True,False,True,False,False,False,False,True,False,True,True,False,False,False,False,False,True,True,True,True,False,True,False,False,False,True,True,True,False,True,False,False,True,True,False,False,False,False,True,False,True,False,False,False,False,True,False,True,False,True,True,False,True,True,True,True,False,False,False,True,True,True,True,True,False,True,True,False,True,False,False,True,False,True,True,True,False,True,False,False,True,True,False,False,True,True,True,True,False,False,False,True,True,True,False,True,False,True,False,False,True,False,True,False,True,False,True,True,True,True,True,False,True,False,False,False,True,False,True,True,False,False,True,True,True,True,False,False,True,False,False,False,True,True,False,False,True,True,False,False,True,True,True,False,True,True,False,True,True,True,False,True,True,True,False,True,True,True,True,False,True,True,False,False,False,True,True,True,False,True,True,True,False,False,True,False,False,True,True,True,True,True,True,False,True,True,True,False,False,False,False,False,True,False,False,True,True,True,False,False,False,False,True,True,False,True,True,False,False,False,False,True,True,True,False,False,False,True,False], dtype = "bool")#candidate|15004|(945,)|const|bool
call_15003 = relay.TupleGetItem(func_13712_call(relay.reshape(const_15004.astype('bool'), [105, 9])), 0)
call_15005 = relay.TupleGetItem(func_13714_call(relay.reshape(const_15004.astype('bool'), [105, 9])), 0)
func_9257_call = mod.get_global_var('func_9257')
func_9258_call = mutated_mod.get_global_var('func_9258')
call_15039 = relay.TupleGetItem(func_9257_call(), 0)
call_15040 = relay.TupleGetItem(func_9258_call(), 0)
func_13803_call = mod.get_global_var('func_13803')
func_13805_call = mutated_mod.get_global_var('func_13805')
call_15043 = func_13803_call()
call_15044 = func_13803_call()
output = relay.Tuple([call_15001,call_15003,const_15004,call_15039,call_15043,])
output2 = relay.Tuple([call_15002,call_15005,const_15004,call_15040,call_15044,])
func_15047 = relay.Function([], output)
mod['func_15047'] = func_15047
mod = relay.transform.InferType()(mod)
mutated_mod['func_15047'] = func_15047
mutated_mod = relay.transform.InferType()(mutated_mod)
func_15047_call = mutated_mod.get_global_var('func_15047')
call_15048 = func_15047_call()
output = call_15048
func_15049 = relay.Function([], output)
mutated_mod['func_15049'] = func_15049
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7461_call = mod.get_global_var('func_7461')
func_7463_call = mutated_mod.get_global_var('func_7463')
call_15062 = relay.TupleGetItem(func_7461_call(), 0)
call_15063 = relay.TupleGetItem(func_7463_call(), 0)
func_11199_call = mod.get_global_var('func_11199')
func_11200_call = mutated_mod.get_global_var('func_11200')
call_15093 = func_11199_call()
call_15094 = func_11199_call()
func_3139_call = mod.get_global_var('func_3139')
func_3143_call = mutated_mod.get_global_var('func_3143')
var_15108 = relay.var("var_15108", dtype = "float64", shape = (75,))#candidate|15108|(75,)|var|float64
var_15109 = relay.var("var_15109", dtype = "float64", shape = (525,))#candidate|15109|(525,)|var|float64
call_15107 = relay.TupleGetItem(func_3139_call(relay.reshape(var_15108.astype('float64'), [5, 15, 1]), relay.reshape(var_15109.astype('float64'), [5, 15, 7]), ), 0)
call_15110 = relay.TupleGetItem(func_3143_call(relay.reshape(var_15108.astype('float64'), [5, 15, 1]), relay.reshape(var_15109.astype('float64'), [5, 15, 7]), ), 0)
output = relay.Tuple([call_15062,call_15093,call_15107,var_15108,var_15109,])
output2 = relay.Tuple([call_15063,call_15094,call_15110,var_15108,var_15109,])
func_15121 = relay.Function([var_15108,var_15109,], output)
mod['func_15121'] = func_15121
mod = relay.transform.InferType()(mod)
mutated_mod['func_15121'] = func_15121
mutated_mod = relay.transform.InferType()(mutated_mod)
func_15121_call = mutated_mod.get_global_var('func_15121')
var_15123 = relay.var("var_15123", dtype = "float64", shape = (75,))#candidate|15123|(75,)|var|float64
var_15124 = relay.var("var_15124", dtype = "float64", shape = (525,))#candidate|15124|(525,)|var|float64
call_15122 = func_15121_call(var_15123,var_15124,)
output = call_15122
func_15125 = relay.Function([var_15123,var_15124,], output)
mutated_mod['func_15125'] = func_15125
mutated_mod = relay.transform.InferType()(mutated_mod)
func_15047_call = mod.get_global_var('func_15047')
func_15049_call = mutated_mod.get_global_var('func_15049')
call_15148 = relay.TupleGetItem(func_15047_call(), 3)
call_15149 = relay.TupleGetItem(func_15049_call(), 3)
func_9750_call = mod.get_global_var('func_9750')
func_9751_call = mutated_mod.get_global_var('func_9751')
call_15176 = func_9750_call()
call_15177 = func_9750_call()
func_8511_call = mod.get_global_var('func_8511')
func_8514_call = mutated_mod.get_global_var('func_8514')
var_15182 = relay.var("var_15182", dtype = "bool", shape = (945,))#candidate|15182|(945,)|var|bool
call_15181 = relay.TupleGetItem(func_8511_call(relay.reshape(var_15182.astype('bool'), [945,])), 0)
call_15183 = relay.TupleGetItem(func_8514_call(relay.reshape(var_15182.astype('bool'), [945,])), 0)
output = relay.Tuple([call_15148,call_15176,call_15181,var_15182,])
output2 = relay.Tuple([call_15149,call_15177,call_15183,var_15182,])
func_15208 = relay.Function([var_15182,], output)
mod['func_15208'] = func_15208
mod = relay.transform.InferType()(mod)
mutated_mod['func_15208'] = func_15208
mutated_mod = relay.transform.InferType()(mutated_mod)
var_15209 = relay.var("var_15209", dtype = "bool", shape = (945,))#candidate|15209|(945,)|var|bool
func_15208_call = mutated_mod.get_global_var('func_15208')
call_15210 = func_15208_call(var_15209)
output = call_15210
func_15211 = relay.Function([var_15209], output)
mutated_mod['func_15211'] = func_15211
mutated_mod = relay.transform.InferType()(mutated_mod)
var_15272 = relay.var("var_15272", dtype = "float64", shape = (9, 6, 2))#candidate|15272|(9, 6, 2)|var|float64
var_15273 = relay.var("var_15273", dtype = "float64", shape = (9, 6, 2))#candidate|15273|(9, 6, 2)|var|float64
bop_15274 = relay.equal(var_15272.astype('bool'), relay.reshape(var_15273.astype('bool'), relay.shape_of(var_15272))) # shape=(9, 6, 2)
output = bop_15274
output2 = bop_15274
func_15284 = relay.Function([var_15272,var_15273,], output)
mod['func_15284'] = func_15284
mod = relay.transform.InferType()(mod)
mutated_mod['func_15284'] = func_15284
mutated_mod = relay.transform.InferType()(mutated_mod)
func_15284_call = mutated_mod.get_global_var('func_15284')
var_15286 = relay.var("var_15286", dtype = "float64", shape = (9, 6, 2))#candidate|15286|(9, 6, 2)|var|float64
var_15287 = relay.var("var_15287", dtype = "float64", shape = (9, 6, 2))#candidate|15287|(9, 6, 2)|var|float64
call_15285 = func_15284_call(var_15286,var_15287,)
output = call_15285
func_15288 = relay.Function([var_15286,var_15287,], output)
mutated_mod['func_15288'] = func_15288
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6146_call = mod.get_global_var('func_6146')
func_6148_call = mutated_mod.get_global_var('func_6148')
call_15301 = func_6146_call()
call_15302 = func_6146_call()
func_7259_call = mod.get_global_var('func_7259')
func_7260_call = mutated_mod.get_global_var('func_7260')
call_15320 = relay.TupleGetItem(func_7259_call(), 0)
call_15321 = relay.TupleGetItem(func_7260_call(), 0)
output = relay.Tuple([call_15301,call_15320,])
output2 = relay.Tuple([call_15302,call_15321,])
func_15323 = relay.Function([], output)
mod['func_15323'] = func_15323
mod = relay.transform.InferType()(mod)
output = func_15323()
func_15324 = relay.Function([], output)
mutated_mod['func_15324'] = func_15324
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6366_call = mod.get_global_var('func_6366')
func_6368_call = mutated_mod.get_global_var('func_6368')
call_15352 = relay.TupleGetItem(func_6366_call(), 0)
call_15353 = relay.TupleGetItem(func_6368_call(), 0)
func_12778_call = mod.get_global_var('func_12778')
func_12779_call = mutated_mod.get_global_var('func_12779')
call_15380 = func_12778_call()
call_15381 = func_12778_call()
output = relay.Tuple([call_15352,call_15380,])
output2 = relay.Tuple([call_15353,call_15381,])
func_15384 = relay.Function([], output)
mod['func_15384'] = func_15384
mod = relay.transform.InferType()(mod)
output = func_15384()
func_15385 = relay.Function([], output)
mutated_mod['func_15385'] = func_15385
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13453_call = mod.get_global_var('func_13453')
func_13455_call = mutated_mod.get_global_var('func_13455')
call_15405 = func_13453_call()
call_15406 = func_13453_call()
func_11412_call = mod.get_global_var('func_11412')
func_11413_call = mutated_mod.get_global_var('func_11413')
call_15407 = relay.TupleGetItem(func_11412_call(), 1)
call_15408 = relay.TupleGetItem(func_11413_call(), 1)
func_6751_call = mod.get_global_var('func_6751')
func_6753_call = mutated_mod.get_global_var('func_6753')
call_15412 = func_6751_call()
call_15413 = func_6751_call()
output = relay.Tuple([call_15405,call_15407,call_15412,])
output2 = relay.Tuple([call_15406,call_15408,call_15413,])
func_15418 = relay.Function([], output)
mod['func_15418'] = func_15418
mod = relay.transform.InferType()(mod)
output = func_15418()
func_15419 = relay.Function([], output)
mutated_mod['func_15419'] = func_15419
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7737_call = mod.get_global_var('func_7737')
func_7738_call = mutated_mod.get_global_var('func_7738')
call_15444 = func_7737_call()
call_15445 = func_7737_call()
func_4185_call = mod.get_global_var('func_4185')
func_4190_call = mutated_mod.get_global_var('func_4190')
const_15447 = relay.const([7.403766,0.531466,-6.917269,-2.555746,2.216085,6.460178,-3.630868,0.021390,-8.965048,0.128906,-0.490782,6.675314,5.282893,7.508013,-5.105852,-5.673442,3.463248,-4.665235,9.878098,6.937790,1.607767,2.410394,9.748827,-5.062833,-2.389313,-0.095487,-5.506117,-1.224392,1.914920,-4.911285,4.908081,-4.432003,3.008557,-2.042617,8.578655,8.526816,5.172756,-3.186193,2.302071,-8.658819,-9.495537,7.898606,-2.849316,-4.016011,7.473138,3.073448,2.120249,-4.132388,-3.982865,-0.263997,-5.821651,4.356554,-1.324784,-7.591731,-3.625781,-4.753829,4.269912,4.561186,1.535557,-8.660809,1.499650,6.205832,3.904861,3.542302,-4.500478,5.487400,7.519353,-7.070954,4.742548,7.716632,7.197495,-6.606176,0.014411,9.471595,-5.343369,-2.792784,4.804214,-0.383881,4.980767,8.597266,1.538572,-3.337339,4.792735,5.853205,-9.932246,-4.553200,-1.173082,-4.384584,5.634266,-7.442052,2.078584,6.213204,-5.575735,8.551229,1.732877,1.748764,-7.836662,5.270565,5.406788,-5.275343,4.637097,8.401084,8.481425,-8.676714,4.539928,-8.506810,8.862428,-4.327365,-8.782228,4.012431,-1.744683,-0.382791,-9.061226,-4.829072,-6.730396,2.436169,3.029268,1.927253,1.829631,1.802180,-6.011749,9.360481,7.595615,3.162319,-8.851379,9.912453,-8.474079,6.399233,4.827298,-9.327840,-3.632643,-7.506428,-1.161735,-7.392049,2.932287,6.179033,-2.714071,1.653398,-6.188672,3.683146,8.958975,5.587985,-1.500158,-8.217620,-3.837909,-7.290270,2.485814,-3.477169,-8.768546,7.165231,3.726650,-3.345888,7.591756,4.336053,-3.693714,-7.384629,-0.484324,3.986332,-0.757784,-4.664476,-7.649488,4.500500,9.275641,-1.994006,5.549244,3.821605,-0.379441,-0.518114,6.928214,4.194741,-5.381542,1.460942,2.316205,0.363593,6.868475,-7.801683,-9.185322,9.266137,2.113113,-2.960868,0.229037,5.655910,3.682087,0.427061,-1.842928,-0.342844,8.862823,6.495582,8.498220,-6.514537,-0.684296,-2.786101,8.725396,7.328279,4.715216,6.263317,4.083014,-3.849450,-0.218014,-5.235011,-6.992683,2.948375,-6.790375,-4.828663,7.193138,-4.405971,4.008095,7.148315,-3.168515,-3.759118,-5.320515,-1.679827,-0.409844,-1.158789,9.173223,4.914538,6.428841,6.827516,6.906661,1.502572,-4.952088,9.254910,8.995303,-8.482034,4.160369,3.122917,-5.647782,1.117398,9.232235,8.378348,-6.403045,-6.651239,1.124145,-6.186894,-7.508352,-2.330837,6.135047,-8.153166,-0.828906,6.795743,2.697651,6.781995,-5.906778,4.752068,-2.405319,-1.638433,9.744559,-4.071582,-8.825938,-4.421917,8.760839,-1.720594,-2.158199,0.203894,7.108837,-2.462576,5.315252,9.084089,-7.411737,0.631023,9.466492,-7.976091,5.270874,-7.195745,4.192843,-1.642091,1.320788,5.111862,7.076840,-7.529600,1.393589,-4.798744,-6.423115,2.488753,-3.521915,7.358355,-7.706363,-8.678417,-5.230419,-5.779664,5.057477,-9.013084,7.597729,-0.735385,-5.362606,8.633269,4.599109,8.204173,4.208363,2.794028,2.770827,-4.993254,-9.812023,1.922540,5.067576,-4.575358,-9.472226,0.103730,-2.619500,1.457834,-4.740597,4.986780,-1.524315,9.069575,0.008463,7.509633,9.754876,-9.089072,-5.138968,-7.902711,8.601576,0.184539,-1.378969,9.018657,-3.905198,2.700220,-2.414077,8.589876,0.360731,-3.832517,9.992514,-5.480420,7.953656,-8.596287,-3.183465,5.794619,7.883202,-2.262509,9.832215,-6.788956,-3.192846,-9.513031,-9.284749,7.330826,-4.593466,-7.676891,8.616581,-9.000856,-5.815196,-7.497538,-5.695439,-8.657965,-5.328678,9.492413,3.683303,-5.638512,5.336013,-5.075992,-7.496019,-4.494476,8.013502,-6.571429,2.489277,-1.379585,3.853965,-7.923580,-7.598512,5.783415,9.475075,4.881901,-0.716824,-4.766952,-2.387441,-4.613131,-4.849607,-6.315805,3.815086,-4.063701,9.755717,-3.778383,-6.396762,-7.876980,-8.674815,2.275330,-1.369550,-1.894083,-1.899628,3.673355,0.800475,2.068509,-3.089762,8.106934,-0.525449,-9.422883,-5.469645,-7.174613,5.454776,-9.714995,-0.191276,-8.967879,-3.582915,-4.251210,0.902666,8.449578,0.625887,6.170721,-2.356832,4.199403,-3.159470,9.911035,1.994283,9.171562,-3.437151,-2.718405,-7.716287,-7.257755,3.580094,5.339686,-3.304299,6.070931,-6.136772,0.372872,7.789906,7.326360,-4.364800,2.382978,3.482141,-7.239113,-4.656167,-1.930940,6.038233,3.830076,0.837584,-0.395530,2.877836,4.406866,-3.902307,9.833570,-7.048881,6.164193,3.273601,-4.145959,-7.573360,-1.550203,7.873281,8.319021,4.997731,3.831969,4.327328,-1.399809,2.285398,-3.128924,-0.605741,9.323965,6.478510,2.002235,1.670108,3.421829,7.486575,-2.577139,-5.004337,6.472501,-8.913631,6.320807,-3.604950,2.755573,7.644710,3.947974,5.497161,-0.925702,-3.718720,7.983804,6.389856,1.048050,-4.934218,4.638216,5.493692,9.677606,8.962923,9.993064,-7.222459,6.744610,6.043768,-6.894874,-6.991120,7.257674,-4.867512,-3.289550,7.419959,-8.755194,-2.786062,-8.872376,0.864359,3.094636,-5.979144,-4.190955,1.047729,7.368796,8.010699,4.819692,-3.379658,-5.762885,1.578154,0.281416,-4.698248,5.295988,2.025621,-3.609197,3.616789,-7.657994,-1.538925,-6.731420,-9.568560,-8.309099,9.324439,6.949016,-6.695159,0.380910,3.734238,-3.377782,-5.471863,-6.938885,-6.471658,-1.849051,-7.879491,6.404526,-6.276169,0.805787,0.104593,-4.150879,5.724017,5.135784,2.815560,-3.453409,7.526248,-0.662869,-0.793545,8.749636,2.989949,-6.210895,-9.481673,-9.100605,5.130567,-0.197021,-3.740912,-0.003620,-0.020183,5.476651,-8.103080,-0.284350,6.029715,1.017661,6.495987,-6.069140,2.093411,9.992116,-8.455159,-8.285583,0.522724,5.669044,5.092347,4.439876,2.531981,3.147397,-8.704682,5.182031,-6.521421,5.725539,6.245667,-1.309391,-7.354981,2.272017,1.925663,1.094938,-8.231413,-3.215649,-1.277652,5.966604,7.047376,9.702208,1.053073,4.195754,6.796107,8.677935,-0.526384,6.628086,4.918376,9.542162,-4.371132,4.441471,9.161231,-5.854514,-1.358870,6.427279,-9.507787,2.193657,8.360932,3.424182,-2.647640,-7.775810,-4.078558,5.206807,-7.007386,3.632794,4.189812,0.396618,-8.081701,-6.784748,3.130367,-9.431465,-6.941168,7.041087,-4.827343,9.633237,-9.876109,6.556120,9.663305,-8.987956,-0.051912,9.237821,6.038071,-1.507581,-7.494398,-5.621606,1.115747,-3.971189,-9.915630,1.842431,-8.140152,3.880830,-6.954619,-8.155534,-8.592124,5.348250,-9.364966,-1.601480,5.595942,6.948223,-7.221481,-3.490869,1.938685,3.931113,-2.415964,-7.843705,4.313055,3.587589,0.681952,-9.674425,1.436188,-5.234808,-9.483195,2.918395,4.102539,3.945155,-4.726734,8.958343,-8.989538,9.312663,-3.682788,-2.070434,-0.466868,9.680115,2.004689,9.732880,-9.815370,-3.345348,-1.809400,-7.559604,-7.101387,9.088839,4.040225,-8.387897,-7.982173,2.965917,-9.079474,-5.311886,4.856694,-6.691441,4.712601,9.968467,6.458810,-7.152220,6.603090,1.562117,3.427118,-4.990714,-2.813872,-9.551577,0.870950,6.937821,5.412694,-5.714587,1.794167,0.031677,-2.310976,-5.592994,-5.147315,0.571601,-4.407723,-9.363273,-6.088558,-5.436553,8.871140,2.084897,-3.184097,-0.182695,-8.192038,0.066311,7.845408,9.188126,5.232334,3.918941,-1.414851,-4.703973,-5.002425,-9.167584,0.695689,6.168900,-4.024453,8.077403,9.310308,-0.636182,-3.206163,7.500395,2.326974,3.098908,5.305977,-1.633469,-4.149670,-6.519826,5.118528,8.600744,6.307466,-2.566101,9.085250,2.620141,0.323417,-8.999425,0.788125,-8.094956,-7.152701,6.475206,-9.305531,-1.507091,-5.946882,3.650145,7.034888,4.403183,7.856952,-6.956312,3.647955,8.293214,-4.575583,0.740906,-1.239487,-9.003138,2.830794,-9.496960,0.028272,-8.072107,5.139214,-3.735048,-7.367685,3.073132,3.606016,-2.578467,5.060614,2.964006,-3.068840,7.941787,-5.217496,3.530633,-6.363865,-7.097407,-9.721091,4.327762,-6.572144,3.262906,2.235245,3.010613,-0.455817,-8.949275,3.232386,3.801441,-2.512999,-4.036671,-3.513360,-9.768224,1.071338,6.344096,-1.606296,2.659119,-1.139735,4.071870,-9.627865,-3.111522,9.519595,-4.017529,-5.172355,-6.788268,9.822005,-9.002919], dtype = "float32")#candidate|15447|(792,)|const|float32
var_15448 = relay.var("var_15448", dtype = "float64", shape = (1280,))#candidate|15448|(1280,)|var|float64
const_15449 = relay.const([[-4,-5,-5,-5,-7,-3,1,9,-10,8,-2,9,-4,4,6,6,-10,1,1,2,-2,-3,7,-3,-10,7,1,-7,10,4,-9,7,3,-7,-3,4,-1,8,-9,-7,2,-3,9,-10,3,-6,-6,4,-1,4,7,7,3,8,-6,2,-8,-1,-7,5,-4,-4,1,8,6,-6,-9,5,-8,-5,7,-2,-2,-9,1,9,-4,-6,1,-7,-7,2,-7,8,-9,2,10,-8,-5,3,2,3,-5,10,-4,-4,3,-7,-2,-6,7,6,1,2,-4,6,-7,1,10,7,-2,5,7,-4,6,-2,9,10,6,8,7,6,-2,3,2,-6,-10,-8,4,10,3,2,-1,-1,-4,-7,-2,-4,-3,4,8,-10,2,7,-9,5,8,5,10,8,-7,-7,-1,-1,8,10,9,6,7,9,-2,-9,2,2,1,7,5,7],[3,-10,5,-8,-1,6,-8,5,7,-10,9,-10,2,-10,-1,4,-5,6,-6,4,7,-5,-5,4,-4,3,-6,6,2,4,-5,7,3,-1,8,-6,5,1,-2,2,-1,1,-9,5,5,5,8,-1,7,-9,7,-1,7,-2,9,4,10,7,-2,-3,2,6,-1,-2,-9,5,-6,-10,-2,-1,8,-3,9,-8,-2,8,8,-5,-3,9,-8,9,3,-7,2,8,-3,6,10,-3,-6,10,6,-5,2,2,8,-4,-1,-8,10,2,-6,7,4,-8,-2,-6,-4,7,3,-8,1,2,4,4,5,-9,-6,-4,-6,-3,-3,-7,8,-4,5,3,-5,-9,-6,-5,1,-9,10,-7,10,-7,9,-1,-9,-9,9,2,9,-1,6,-4,-9,4,-10,6,-3,4,-6,-8,-2,-1,10,-2,-7,-4,10,2,4,-9,-10,-6]], dtype = "uint16")#candidate|15449|(2, 168)|const|uint16
call_15446 = relay.TupleGetItem(func_4185_call(relay.reshape(const_15447.astype('float32'), [8, 11, 9]), relay.reshape(var_15448.astype('float64'), [1280,]), relay.reshape(const_15449.astype('uint16'), [336,]), ), 5)
call_15450 = relay.TupleGetItem(func_4190_call(relay.reshape(const_15447.astype('float32'), [8, 11, 9]), relay.reshape(var_15448.astype('float64'), [1280,]), relay.reshape(const_15449.astype('uint16'), [336,]), ), 5)
output = relay.Tuple([call_15444,call_15446,const_15447,var_15448,const_15449,])
output2 = relay.Tuple([call_15445,call_15450,const_15447,var_15448,const_15449,])
func_15459 = relay.Function([var_15448,], output)
mod['func_15459'] = func_15459
mod = relay.transform.InferType()(mod)
mutated_mod['func_15459'] = func_15459
mutated_mod = relay.transform.InferType()(mutated_mod)
var_15460 = relay.var("var_15460", dtype = "float64", shape = (1280,))#candidate|15460|(1280,)|var|float64
func_15459_call = mutated_mod.get_global_var('func_15459')
call_15461 = func_15459_call(var_15460)
output = call_15461
func_15462 = relay.Function([var_15460], output)
mutated_mod['func_15462'] = func_15462
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13045_call = mod.get_global_var('func_13045')
func_13046_call = mutated_mod.get_global_var('func_13046')
call_15466 = relay.TupleGetItem(func_13045_call(), 0)
call_15467 = relay.TupleGetItem(func_13046_call(), 0)
func_8578_call = mod.get_global_var('func_8578')
func_8579_call = mutated_mod.get_global_var('func_8579')
call_15485 = relay.TupleGetItem(func_8578_call(), 0)
call_15486 = relay.TupleGetItem(func_8579_call(), 0)
func_9257_call = mod.get_global_var('func_9257')
func_9258_call = mutated_mod.get_global_var('func_9258')
call_15489 = relay.TupleGetItem(func_9257_call(), 1)
call_15490 = relay.TupleGetItem(func_9258_call(), 1)
output = relay.Tuple([call_15466,call_15485,call_15489,])
output2 = relay.Tuple([call_15467,call_15486,call_15490,])
func_15493 = relay.Function([], output)
mod['func_15493'] = func_15493
mod = relay.transform.InferType()(mod)
output = func_15493()
func_15494 = relay.Function([], output)
mutated_mod['func_15494'] = func_15494
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11259_call = mod.get_global_var('func_11259')
func_11261_call = mutated_mod.get_global_var('func_11261')
call_15594 = relay.TupleGetItem(func_11259_call(), 1)
call_15595 = relay.TupleGetItem(func_11261_call(), 1)
func_12820_call = mod.get_global_var('func_12820')
func_12821_call = mutated_mod.get_global_var('func_12821')
call_15597 = relay.TupleGetItem(func_12820_call(), 2)
call_15598 = relay.TupleGetItem(func_12821_call(), 2)
output = relay.Tuple([call_15594,call_15597,])
output2 = relay.Tuple([call_15595,call_15598,])
func_15606 = relay.Function([], output)
mod['func_15606'] = func_15606
mod = relay.transform.InferType()(mod)
mutated_mod['func_15606'] = func_15606
mutated_mod = relay.transform.InferType()(mutated_mod)
func_15606_call = mutated_mod.get_global_var('func_15606')
call_15607 = func_15606_call()
output = call_15607
func_15608 = relay.Function([], output)
mutated_mod['func_15608'] = func_15608
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13149_call = mod.get_global_var('func_13149')
func_13151_call = mutated_mod.get_global_var('func_13151')
call_15614 = relay.TupleGetItem(func_13149_call(), 0)
call_15615 = relay.TupleGetItem(func_13151_call(), 0)
func_13987_call = mod.get_global_var('func_13987')
func_13988_call = mutated_mod.get_global_var('func_13988')
call_15628 = func_13987_call()
call_15629 = func_13987_call()
func_13712_call = mod.get_global_var('func_13712')
func_13714_call = mutated_mod.get_global_var('func_13714')
var_15638 = relay.var("var_15638", dtype = "bool", shape = (945,))#candidate|15638|(945,)|var|bool
call_15637 = relay.TupleGetItem(func_13712_call(relay.reshape(var_15638.astype('bool'), [105, 9])), 0)
call_15639 = relay.TupleGetItem(func_13714_call(relay.reshape(var_15638.astype('bool'), [105, 9])), 0)
func_10070_call = mod.get_global_var('func_10070')
func_10072_call = mutated_mod.get_global_var('func_10072')
call_15640 = relay.TupleGetItem(func_10070_call(), 1)
call_15641 = relay.TupleGetItem(func_10072_call(), 1)
output = relay.Tuple([call_15614,call_15628,call_15637,var_15638,call_15640,])
output2 = relay.Tuple([call_15615,call_15629,call_15639,var_15638,call_15641,])
func_15644 = relay.Function([var_15638,], output)
mod['func_15644'] = func_15644
mod = relay.transform.InferType()(mod)
var_15645 = relay.var("var_15645", dtype = "bool", shape = (945,))#candidate|15645|(945,)|var|bool
output = func_15644(var_15645)
func_15646 = relay.Function([var_15645], output)
mutated_mod['func_15646'] = func_15646
mutated_mod = relay.transform.InferType()(mutated_mod)
func_7259_call = mod.get_global_var('func_7259')
func_7260_call = mutated_mod.get_global_var('func_7260')
call_15740 = relay.TupleGetItem(func_7259_call(), 0)
call_15741 = relay.TupleGetItem(func_7260_call(), 0)
func_7136_call = mod.get_global_var('func_7136')
func_7137_call = mutated_mod.get_global_var('func_7137')
call_15746 = func_7136_call()
call_15747 = func_7136_call()
func_6517_call = mod.get_global_var('func_6517')
func_6519_call = mutated_mod.get_global_var('func_6519')
call_15748 = relay.TupleGetItem(func_6517_call(), 0)
call_15749 = relay.TupleGetItem(func_6519_call(), 0)
output = relay.Tuple([call_15740,call_15746,call_15748,])
output2 = relay.Tuple([call_15741,call_15747,call_15749,])
func_15758 = relay.Function([], output)
mod['func_15758'] = func_15758
mod = relay.transform.InferType()(mod)
mutated_mod['func_15758'] = func_15758
mutated_mod = relay.transform.InferType()(mutated_mod)
func_15758_call = mutated_mod.get_global_var('func_15758')
call_15759 = func_15758_call()
output = call_15759
func_15760 = relay.Function([], output)
mutated_mod['func_15760'] = func_15760
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10133_call = mod.get_global_var('func_10133')
func_10134_call = mutated_mod.get_global_var('func_10134')
call_15763 = relay.TupleGetItem(func_10133_call(), 0)
call_15764 = relay.TupleGetItem(func_10134_call(), 0)
output = relay.Tuple([call_15763,])
output2 = relay.Tuple([call_15764,])
func_15765 = relay.Function([], output)
mod['func_15765'] = func_15765
mod = relay.transform.InferType()(mod)
output = func_15765()
func_15766 = relay.Function([], output)
mutated_mod['func_15766'] = func_15766
mutated_mod = relay.transform.InferType()(mutated_mod)
var_15770 = relay.var("var_15770", dtype = "uint8", shape = ())#candidate|15770|()|var|uint8
const_15771 = relay.const([[[-6,8,-3,-5,-2,-1,3,3,-7,7],[8,4,1,9,-8,8,6,-1,6,-1],[-8,-7,10,-8,2,-10,1,4,-7,-3],[-6,6,-9,-2,-4,-5,7,-3,6,5],[7,8,-1,-1,5,-1,7,2,6,-2],[9,3,1,-3,7,5,-3,-1,7,-5]]], dtype = "uint8")#candidate|15771|(1, 6, 10)|const|uint8
bop_15772 = relay.bitwise_and(var_15770.astype('uint8'), const_15771.astype('uint8')) # shape=(1, 6, 10)
output = bop_15772
output2 = bop_15772
func_15782 = relay.Function([var_15770,], output)
mod['func_15782'] = func_15782
mod = relay.transform.InferType()(mod)
var_15783 = relay.var("var_15783", dtype = "uint8", shape = ())#candidate|15783|()|var|uint8
output = func_15782(var_15783)
func_15784 = relay.Function([var_15783], output)
mutated_mod['func_15784'] = func_15784
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10552_call = mod.get_global_var('func_10552')
func_10554_call = mutated_mod.get_global_var('func_10554')
call_15808 = func_10552_call()
call_15809 = func_10552_call()
output = relay.Tuple([call_15808,])
output2 = relay.Tuple([call_15809,])
func_15820 = relay.Function([], output)
mod['func_15820'] = func_15820
mod = relay.transform.InferType()(mod)
mutated_mod['func_15820'] = func_15820
mutated_mod = relay.transform.InferType()(mutated_mod)
func_15820_call = mutated_mod.get_global_var('func_15820')
call_15821 = func_15820_call()
output = call_15821
func_15822 = relay.Function([], output)
mutated_mod['func_15822'] = func_15822
mutated_mod = relay.transform.InferType()(mutated_mod)
var_15823 = relay.var("var_15823", dtype = "float32", shape = (1, 7, 8))#candidate|15823|(1, 7, 8)|var|float32
uop_15824 = relay.sin(var_15823.astype('float32')) # shape=(1, 7, 8)
uop_15826 = relay.acos(uop_15824.astype('float32')) # shape=(1, 7, 8)
output = relay.Tuple([uop_15826,])
output2 = relay.Tuple([uop_15826,])
func_15830 = relay.Function([var_15823,], output)
mod['func_15830'] = func_15830
mod = relay.transform.InferType()(mod)
var_15831 = relay.var("var_15831", dtype = "float32", shape = (1, 7, 8))#candidate|15831|(1, 7, 8)|var|float32
output = func_15830(var_15831)
func_15832 = relay.Function([var_15831], output)
mutated_mod['func_15832'] = func_15832
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11140_call = mod.get_global_var('func_11140')
func_11142_call = mutated_mod.get_global_var('func_11142')
call_15858 = relay.TupleGetItem(func_11140_call(), 0)
call_15859 = relay.TupleGetItem(func_11142_call(), 0)
output = call_15858
output2 = call_15859
func_15860 = relay.Function([], output)
mod['func_15860'] = func_15860
mod = relay.transform.InferType()(mod)
mutated_mod['func_15860'] = func_15860
mutated_mod = relay.transform.InferType()(mutated_mod)
func_15860_call = mutated_mod.get_global_var('func_15860')
call_15861 = func_15860_call()
output = call_15861
func_15862 = relay.Function([], output)
mutated_mod['func_15862'] = func_15862
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6880_call = mod.get_global_var('func_6880')
func_6881_call = mutated_mod.get_global_var('func_6881')
call_15884 = relay.TupleGetItem(func_6880_call(), 0)
call_15885 = relay.TupleGetItem(func_6881_call(), 0)
output = call_15884
output2 = call_15885
func_15900 = relay.Function([], output)
mod['func_15900'] = func_15900
mod = relay.transform.InferType()(mod)
output = func_15900()
func_15901 = relay.Function([], output)
mutated_mod['func_15901'] = func_15901
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8428_call = mod.get_global_var('func_8428')
func_8429_call = mutated_mod.get_global_var('func_8429')
call_16019 = relay.TupleGetItem(func_8428_call(), 0)
call_16020 = relay.TupleGetItem(func_8429_call(), 0)
func_14223_call = mod.get_global_var('func_14223')
func_14225_call = mutated_mod.get_global_var('func_14225')
call_16049 = relay.TupleGetItem(func_14223_call(), 0)
call_16050 = relay.TupleGetItem(func_14225_call(), 0)
func_14063_call = mod.get_global_var('func_14063')
func_14064_call = mutated_mod.get_global_var('func_14064')
call_16056 = relay.TupleGetItem(func_14063_call(), 0)
call_16057 = relay.TupleGetItem(func_14064_call(), 0)
func_13920_call = mod.get_global_var('func_13920')
func_13921_call = mutated_mod.get_global_var('func_13921')
call_16063 = relay.TupleGetItem(func_13920_call(), 0)
call_16064 = relay.TupleGetItem(func_13921_call(), 0)
func_8121_call = mod.get_global_var('func_8121')
func_8123_call = mutated_mod.get_global_var('func_8123')
call_16070 = func_8121_call()
call_16071 = func_8121_call()
var_16076 = relay.var("var_16076", dtype = "bool", shape = (945,))#candidate|16076|(945,)|var|bool
bop_16077 = relay.left_shift(call_16056.astype('uint8'), relay.reshape(var_16076.astype('uint8'), relay.shape_of(call_16056))) # shape=(945,)
bop_16080 = relay.left_shift(call_16057.astype('uint8'), relay.reshape(var_16076.astype('uint8'), relay.shape_of(call_16057))) # shape=(945,)
bop_16083 = relay.divide(var_16076.astype('float32'), relay.reshape(call_16056.astype('float32'), relay.shape_of(var_16076))) # shape=(945,)
bop_16086 = relay.divide(var_16076.astype('float32'), relay.reshape(call_16057.astype('float32'), relay.shape_of(var_16076))) # shape=(945,)
func_9004_call = mod.get_global_var('func_9004')
func_9008_call = mutated_mod.get_global_var('func_9008')
var_16095 = relay.var("var_16095", dtype = "float32", shape = (6, 132))#candidate|16095|(6, 132)|var|float32
call_16094 = relay.TupleGetItem(func_9004_call(relay.reshape(var_16095.astype('float32'), [792,]), relay.reshape(call_16019.astype('float64'), [1280,]), ), 1)
call_16096 = relay.TupleGetItem(func_9008_call(relay.reshape(var_16095.astype('float32'), [792,]), relay.reshape(call_16019.astype('float64'), [1280,]), ), 1)
const_16099 = relay.const([[[-6.885191,3.166791,0.340271,-8.469116,-9.770608,-8.852516,-9.687668,-0.150753,0.764558,9.087520],[7.693423,-0.856169,-4.186127,9.427463,7.576488,-0.234565,-5.707522,9.069544,-4.982929,2.357714]],[[3.855968,-6.572045,-7.674745,-4.475351,6.725687,3.412032,-8.036290,9.054522,3.359842,3.843532],[-3.352232,-9.050976,5.670927,-7.239172,6.231582,3.105691,5.451021,-4.765768,-2.862715,9.960643]],[[9.225171,-7.841767,9.028333,-0.609653,5.137643,2.500117,-0.033653,-4.947118,1.218150,9.983962],[-4.535992,7.816158,-1.287261,9.132961,3.384139,-7.470728,1.956604,-0.974300,-5.283774,6.294861]],[[-9.166046,9.038353,8.113239,0.787360,-9.468501,8.769780,2.777250,-8.644322,2.517152,-5.805717],[2.811091,2.445044,1.075298,0.612546,-9.341125,9.691412,9.817120,1.881902,-6.306888,7.318934]],[[9.944767,-4.683253,3.014292,-6.754713,-2.846169,7.205948,8.530009,5.592407,6.400207,6.972221],[4.370461,-5.438053,4.199309,1.608892,-1.967608,7.694069,-8.548480,-9.282367,-3.762093,-4.902393]],[[9.699885,-0.449873,-8.057068,-2.871467,-7.765623,4.428639,-2.595502,-9.234357,-7.979923,-3.499752],[-9.848416,-0.372665,5.648473,7.365621,-5.305248,-9.077761,1.349230,-3.120191,-4.369457,5.444150]],[[-4.159190,1.714672,8.291432,9.413788,6.093419,-1.492364,2.199014,-9.694500,-9.698686,-9.080880],[5.770771,-7.864651,-2.120996,-2.071764,2.242253,9.342179,4.302422,4.011518,9.104680,-8.905475]],[[-1.984253,-1.410487,1.658753,-3.262548,-1.548553,-2.290585,4.038548,7.748047,-0.979707,7.378735],[1.265960,5.144973,5.545882,2.574118,8.164367,6.475762,4.512670,5.113026,6.992357,-5.414416]],[[8.400494,7.198577,6.401048,-7.508329,4.855119,-0.190010,-4.271346,7.220892,-2.257099,-3.086240],[-5.672060,2.567449,-4.470789,-7.256735,-3.974783,9.268806,-3.387027,-7.881936,7.363455,6.465648]],[[-7.835636,-3.194248,6.155781,-9.066260,-1.087144,-8.338541,1.770613,-1.358261,-7.406825,3.456330],[0.151472,-7.924691,7.508635,-6.921457,5.425964,9.439285,2.919272,0.044785,5.980165,-4.926738]],[[-6.499278,-7.933196,-1.766084,5.201352,-0.531126,9.598418,-4.862786,8.748289,-4.422121,-7.746337],[4.922803,-1.867493,9.093108,1.708516,6.271225,1.379379,-0.954268,0.701554,-2.417656,-8.262105]],[[7.428124,2.674783,7.596394,-4.947753,-1.339278,-8.805832,-3.610907,-6.331874,-5.647118,2.376475],[8.583074,6.982891,-6.101568,-2.069986,-0.423582,2.127011,-3.076853,2.666799,1.146355,5.615010]],[[-9.360406,0.923945,-7.082490,7.410456,-7.633033,4.209965,4.626178,8.033498,7.504655,-2.676879],[-1.810412,-4.688235,-8.534414,5.418279,2.249498,6.321449,-0.719864,-9.939196,-5.286394,5.684014]],[[-2.465608,-8.550705,5.981911,-0.015677,1.060205,-6.836497,-5.082726,-9.287722,-1.219264,0.190728],[9.684794,-3.671098,0.570840,-5.914289,-0.560901,-7.553947,-4.243947,-5.410475,3.531345,7.301259]]], dtype = "float64")#candidate|16099|(14, 2, 10)|const|float64
bop_16100 = relay.bitwise_or(call_16070.astype('int16'), relay.reshape(const_16099.astype('int16'), relay.shape_of(call_16070))) # shape=(14, 2, 10)
bop_16103 = relay.bitwise_or(call_16071.astype('int16'), relay.reshape(const_16099.astype('int16'), relay.shape_of(call_16071))) # shape=(14, 2, 10)
func_5143_call = mod.get_global_var('func_5143')
func_5144_call = mutated_mod.get_global_var('func_5144')
call_16109 = relay.TupleGetItem(func_5143_call(), 2)
call_16110 = relay.TupleGetItem(func_5144_call(), 2)
output = relay.Tuple([call_16019,call_16049,call_16063,bop_16077,bop_16083,call_16094,var_16095,bop_16100,call_16109,])
output2 = relay.Tuple([call_16020,call_16050,call_16064,bop_16080,bop_16086,call_16096,var_16095,bop_16103,call_16110,])
func_16114 = relay.Function([var_16076,var_16095,], output)
mod['func_16114'] = func_16114
mod = relay.transform.InferType()(mod)
var_16115 = relay.var("var_16115", dtype = "bool", shape = (945,))#candidate|16115|(945,)|var|bool
var_16116 = relay.var("var_16116", dtype = "float32", shape = (6, 132))#candidate|16116|(6, 132)|var|float32
output = func_16114(var_16115,var_16116,)
func_16117 = relay.Function([var_16115,var_16116,], output)
mutated_mod['func_16117'] = func_16117
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13116_call = mod.get_global_var('func_13116')
func_13118_call = mutated_mod.get_global_var('func_13118')
call_16133 = relay.TupleGetItem(func_13116_call(), 0)
call_16134 = relay.TupleGetItem(func_13118_call(), 0)
output = relay.Tuple([call_16133,])
output2 = relay.Tuple([call_16134,])
func_16135 = relay.Function([], output)
mod['func_16135'] = func_16135
mod = relay.transform.InferType()(mod)
output = func_16135()
func_16136 = relay.Function([], output)
mutated_mod['func_16136'] = func_16136
mutated_mod = relay.transform.InferType()(mutated_mod)
var_16154 = relay.var("var_16154", dtype = "float32", shape = (15, 14, 3))#candidate|16154|(15, 14, 3)|var|float32
uop_16155 = relay.rsqrt(var_16154.astype('float32')) # shape=(15, 14, 3)
func_6631_call = mod.get_global_var('func_6631')
func_6633_call = mutated_mod.get_global_var('func_6633')
const_16159 = relay.const([3.599191,5.447887,3.126675,7.168891,-2.406917,9.830645,1.785156,4.944594,4.321341,4.057070,-4.101869,6.599665,-1.336466,0.897369,7.256550,8.801014,-2.198964,5.048307,-3.974632,-0.678327,-3.056228,-9.892777,4.053723,-3.914322,6.281621,3.099060,3.697524,4.489371,-7.006017,3.536688,-6.175013,4.926338,8.556442,-2.178233,3.295573,-4.159210,8.072474,2.604930,-9.856474,4.996925,4.317538,3.067880,2.011849,-4.463659,7.554375,9.790771,4.682777,-0.213779,7.233415,-3.048226,-4.044205,-4.079625,8.951984,-6.987307,4.533771,-5.333724,2.798308,0.468285,-6.927878,-9.961470,4.616899,9.568671,-0.677383,4.268140,5.557798,-8.064092,-0.939603,0.168899,-5.773555,0.675161,9.370274,8.285715,-6.690607,9.854941,-8.427149,-2.513634,-9.032004,-3.438751,-9.784887,-8.698975,-3.679698,9.408773,-0.194862,-6.033782,-2.447199,-6.980665,3.474128,-4.481856,1.495914,3.439900,-1.974409,9.159330,1.425589,0.784761,-3.734647,1.610532,2.580118,7.807012,4.441636,-4.083211,-9.029646,7.084410,-2.427393,9.374538,-7.967145,-0.570682,-9.318295,3.755408,6.039648,0.204343,-7.422809,-7.652326,8.692749,2.683205,-5.461579,2.898192,0.899024,-8.464733,-1.640922,2.056371,-6.710534,2.756435,5.741258,5.466972,-9.375129,-8.379685,3.498356,-4.548002,6.810846,-3.475023,5.825687,-2.762505,-9.078367,5.197042,-9.010246,0.467301,-8.605552,0.058208,2.533680,-5.792701,-6.779988,5.132061,-9.052866,-3.559594,9.891780,3.699706,-2.812081,9.416202,-8.509938,-9.273529,0.595544,-0.078171,-1.969345,-4.500213,-5.724392,8.446394,2.396772,-4.913426,-6.552203,9.902354,2.358835,4.642496,-0.306597,0.271939,8.080479,8.648609,7.691760,-5.673510,-8.358311,-1.613500,9.697956,5.410635,-2.546719,-4.438806,-2.653109,-1.869236,-9.138470,-3.407032,0.139254,-7.971092,4.684492,8.599572,8.430578,-9.922888,-8.210315,-3.874371,3.159408,8.120363,0.443491,-6.945366,9.009693,-5.920337,-9.052003,7.253211,-0.969805,-3.983116,2.333230,-6.431047,6.878748,-1.682034,9.858801,-5.672299,-0.151910,-5.931064,3.868624,-8.912409,-8.998456,0.096672,7.268889,-6.359265,-2.714106,3.809098,-2.942593,-5.125162,-8.406559,-9.297791,8.490248,-3.197341,-4.679721,9.760757,-6.956684,-6.519066,-9.511891,4.793561,-1.803142,5.819867,3.677069,9.942797,9.756125,0.021318,-0.654946,4.062319,-7.698431,-7.462226,3.946282,9.337252,6.946547,-3.916841,7.693745,3.902212,9.108650,-9.480562,1.107155,-6.547417,0.319607,-7.084846,2.734597,2.845920,-8.661511,-9.245238,7.813185,1.249697,-4.716919,6.382444,9.564979,7.610388,1.475572,2.204048,5.428857,9.957432,-9.633949,2.574235,-1.277171,1.099037,-8.036027,-9.231309,5.518745,-5.546465,1.752659,6.150170,-3.842983,3.425979,-9.711507,-4.639855,-7.734556,-3.020445,7.788848,5.482146,-5.529145,-4.416112,-4.968841,3.032720,-7.059739,9.738338,5.635710,4.526592,-8.821625,3.232365,5.012964,-1.937620,-7.591292,-5.421070,-9.747847,5.868327,4.698170,6.618021,-8.951318,3.410045,-6.149300,-4.574096,-7.861082,0.120948,-5.523647,7.481675,-1.216367,2.852318,3.428034,3.559706,2.920920,-1.045318,-9.442916,8.855956,-1.919323,-4.933069,9.303926,-7.753737,-3.334975,-2.642770,2.563505,-9.987797,-7.981215,-0.702243,-8.694861,5.786559,-0.936242,-0.117950,-8.864847,4.635099,-4.800294,4.568399,-4.099293,1.816691,-4.677099,2.163899,-4.218912,1.499006,5.355274,-5.083352,0.312857,-5.239804,2.835812,6.512938,-9.735138,6.050078,-7.998630,-0.678740,-0.528454,7.428108,-4.856966,-4.745849,-8.398676,4.797981,3.371642,6.034447,5.661186,-1.496110,-7.524205,-1.972909,-7.280296,-0.466817,4.512109,5.641121,-1.438640,-7.628738,6.962874,2.667233,-5.396064,-3.314727,-0.555948,9.123184,1.924382,4.346994,2.959251,-1.212752,4.168188,-6.472724,6.942578,-3.732855,-9.549817,-8.961356,-1.979158,7.063148,-0.645021,-6.059591,-8.857194,1.458354,-0.953768,7.993210,-6.301423,9.778914,-9.800150,-4.922981,-5.253403,-3.074029,8.915619,-1.506269,-5.686399,5.781197,1.258994,-5.299071,5.306622,7.265663,-5.147096,8.616357,-9.394072,-2.113367,-6.994930,9.027433,1.179752,6.304239,-4.780342,-6.237615,2.847795,-4.645047,-3.472890,-5.719477,-9.103181,5.493310,9.912844,1.328536,4.840930,-6.572340,1.365041,-2.366774,3.029625,8.907628,-0.182594,4.038338,2.533196,-9.963468,-4.416696,5.072883,6.550426,4.913940,1.442724,-4.193411,-1.436016,-0.666725,-7.325323,-7.547427,-4.247532,-7.055325,-3.921558,1.793961,9.847601,2.108905,-2.066070,-5.528580,0.203461,-9.995364,-1.412777,3.954366,-3.137650,-9.998613,1.382880,-5.342482,5.725987,-5.029635,-2.705199,2.766155,1.650298,5.137425,-2.474703,5.551684,3.716687,-8.044386,8.699949,-2.654548,-2.043043,-6.164947,-6.655334,4.284749,6.932673,7.137785,2.651745,8.692203,8.775425,8.855256,-5.893761,4.440801,-7.402364,-3.379690,0.967200,0.046364,9.143400,6.687621,2.222731,1.483427,-9.900814,-5.162521,-9.358717,3.700141,1.038577,8.682603,8.563409,4.849619,-6.340978,0.118682,1.981896,4.764372,9.156007,-9.780763,-2.604099,6.046757,-8.511870,-6.775886,-1.710877,-7.068937,-3.227273,0.893849,-6.878854,-9.638266,-3.147873,-8.586793,-9.218093,2.390330,0.455956,-5.245272,9.296616,-2.191899,-3.275915,-5.751818,-8.283977,-7.675554,-9.099384], dtype = "float64")#candidate|16159|(525,)|const|float64
call_16158 = relay.TupleGetItem(func_6631_call(relay.reshape(const_16159.astype('float64'), [35, 15])), 1)
call_16160 = relay.TupleGetItem(func_6633_call(relay.reshape(const_16159.astype('float64'), [35, 15])), 1)
output = relay.Tuple([uop_16155,call_16158,const_16159,])
output2 = relay.Tuple([uop_16155,call_16160,const_16159,])
func_16190 = relay.Function([var_16154,], output)
mod['func_16190'] = func_16190
mod = relay.transform.InferType()(mod)
mutated_mod['func_16190'] = func_16190
mutated_mod = relay.transform.InferType()(mutated_mod)
var_16191 = relay.var("var_16191", dtype = "float32", shape = (15, 14, 3))#candidate|16191|(15, 14, 3)|var|float32
func_16190_call = mutated_mod.get_global_var('func_16190')
call_16192 = func_16190_call(var_16191)
output = call_16192
func_16193 = relay.Function([var_16191], output)
mutated_mod['func_16193'] = func_16193
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8139_call = mod.get_global_var('func_8139')
func_8141_call = mutated_mod.get_global_var('func_8141')
call_16197 = relay.TupleGetItem(func_8139_call(), 0)
call_16198 = relay.TupleGetItem(func_8141_call(), 0)
output = relay.Tuple([call_16197,])
output2 = relay.Tuple([call_16198,])
func_16232 = relay.Function([], output)
mod['func_16232'] = func_16232
mod = relay.transform.InferType()(mod)
output = func_16232()
func_16233 = relay.Function([], output)
mutated_mod['func_16233'] = func_16233
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13920_call = mod.get_global_var('func_13920')
func_13921_call = mutated_mod.get_global_var('func_13921')
call_16237 = relay.TupleGetItem(func_13920_call(), 0)
call_16238 = relay.TupleGetItem(func_13921_call(), 0)
output = call_16237
output2 = call_16238
func_16239 = relay.Function([], output)
mod['func_16239'] = func_16239
mod = relay.transform.InferType()(mod)
mutated_mod['func_16239'] = func_16239
mutated_mod = relay.transform.InferType()(mutated_mod)
func_16239_call = mutated_mod.get_global_var('func_16239')
call_16240 = func_16239_call()
output = call_16240
func_16241 = relay.Function([], output)
mutated_mod['func_16241'] = func_16241
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13569_call = mod.get_global_var('func_13569')
func_13571_call = mutated_mod.get_global_var('func_13571')
call_16253 = relay.TupleGetItem(func_13569_call(), 1)
call_16254 = relay.TupleGetItem(func_13571_call(), 1)
func_6666_call = mod.get_global_var('func_6666')
func_6668_call = mutated_mod.get_global_var('func_6668')
call_16268 = relay.TupleGetItem(func_6666_call(), 0)
call_16269 = relay.TupleGetItem(func_6668_call(), 0)
func_6366_call = mod.get_global_var('func_6366')
func_6368_call = mutated_mod.get_global_var('func_6368')
call_16273 = relay.TupleGetItem(func_6366_call(), 0)
call_16274 = relay.TupleGetItem(func_6368_call(), 0)
func_7976_call = mod.get_global_var('func_7976')
func_7977_call = mutated_mod.get_global_var('func_7977')
call_16275 = relay.TupleGetItem(func_7976_call(), 0)
call_16276 = relay.TupleGetItem(func_7977_call(), 0)
output = relay.Tuple([call_16253,call_16268,call_16273,call_16275,])
output2 = relay.Tuple([call_16254,call_16269,call_16274,call_16276,])
func_16288 = relay.Function([], output)
mod['func_16288'] = func_16288
mod = relay.transform.InferType()(mod)
output = func_16288()
func_16289 = relay.Function([], output)
mutated_mod['func_16289'] = func_16289
mutated_mod = relay.transform.InferType()(mutated_mod)
const_16313 = relay.const([[[4.698157,-4.094774,-1.951711,-0.961022,-1.598204,-2.697281,8.938267,3.013558,9.353600,3.788105],[-9.507478,5.718496,6.374651,1.525696,-7.305014,7.412741,-6.662682,9.363682,-5.544582,-0.385284],[0.014401,1.354521,2.310815,-6.848085,3.479054,-7.186016,4.620443,7.324692,2.419385,-5.470571],[5.483105,2.136360,-2.137455,-3.592032,-3.067196,2.687041,3.804793,0.132703,1.930876,2.470779],[8.624518,6.060247,9.700223,-2.783272,3.186705,9.642747,-5.982192,2.507949,4.188594,2.640260],[-4.820001,0.809328,0.305032,1.280199,-9.955195,6.048093,-3.741364,-1.780922,-6.988465,-8.240879],[-0.340266,-3.885896,-3.530742,8.659349,-5.557379,0.688785,9.373335,-5.788127,3.162299,3.606666],[0.909020,-3.027910,9.187670,5.054766,6.236717,-6.756075,3.366094,-5.945314,-7.752412,-2.136724]],[[-5.119898,1.121381,-0.076188,-4.697184,5.724318,-9.169158,9.023565,-9.830086,0.491889,-4.389979],[4.569893,-4.966998,3.374467,-7.950229,-1.683717,-1.263461,2.756668,-2.629567,-3.719268,0.383579],[5.123013,-5.109003,-6.317577,2.756730,-2.238937,-0.803961,-8.502838,3.471339,5.070354,2.496499],[-7.068147,-2.979953,-4.156352,8.685623,-1.048697,-9.964595,-9.272655,-2.892517,-2.271972,9.744926],[2.941474,-3.838382,-6.183398,6.280893,-2.816872,-4.491666,6.561535,-8.191151,8.702868,-9.749995],[-6.015916,4.621119,3.780513,-7.319284,-4.461217,-7.675796,2.137605,5.326786,-6.752094,-1.432833],[-4.095293,7.120669,1.435306,1.826447,-7.268017,-3.187704,-3.107991,7.788791,3.614234,7.944910],[7.510313,7.986533,6.866985,9.650437,-3.799716,-4.598729,-8.566618,-2.427613,5.555448,-8.548465]],[[8.016866,-0.222410,8.797538,9.847435,6.096548,-3.280467,-9.482959,-6.780815,-1.641245,-6.290260],[3.927325,5.907070,-2.839812,4.142233,8.684739,4.928389,3.344791,5.987444,-7.916082,6.301430],[4.741990,-8.897340,-9.443564,5.139332,-6.114307,1.224897,-3.988647,1.830745,-6.590145,-6.292047],[5.480524,-4.444593,-9.593509,7.670655,-4.570512,-7.895537,-7.773032,8.843217,9.496169,0.908344],[-0.136662,1.197434,-3.824433,-0.416078,-3.265809,-5.848035,-3.931540,-3.282696,-8.988545,1.924807],[-0.100944,3.848621,1.699023,5.289467,6.752882,-1.728948,7.556069,0.356187,9.589266,-5.389300],[-5.179605,8.080889,2.455660,3.896830,8.035992,-0.685448,6.749424,-8.563267,3.928084,5.456971],[-2.355033,-6.816519,-7.902360,3.560512,3.919185,0.636698,3.767438,8.395220,-5.313423,-1.744168]],[[0.569871,2.779745,-0.324542,9.355729,-4.279484,2.365984,1.759036,-4.300964,8.874357,5.369912],[-1.427304,-7.226806,5.337878,-7.453214,8.530955,3.687088,-6.040614,3.334651,-7.670315,-6.252099],[7.787384,9.710774,7.439997,-5.266995,-9.864602,-9.359090,-4.377168,-8.935442,-6.260407,0.736315],[7.338392,-8.317252,-1.195888,7.062867,8.678667,5.967731,7.354097,3.702384,-1.939397,4.267658],[4.377559,0.508391,-9.189424,9.044774,6.545931,-2.449521,8.900487,-7.622990,-4.568445,-3.579999],[0.629343,9.830868,-4.726047,0.943103,-3.253606,4.644507,-8.097262,-5.748612,-2.089949,2.297393],[-4.629327,8.315254,2.382433,3.954238,9.069505,5.243238,8.731885,3.199367,3.930825,3.404322],[8.951245,2.437815,6.492113,-0.387082,5.886767,8.831733,-7.500381,3.145796,-0.103117,7.556736]],[[-1.502054,-8.262186,9.813082,9.376405,4.899251,1.802801,-0.802396,-1.001777,-9.685326,4.128820],[3.498435,3.081662,9.460053,2.653529,-4.023825,7.853750,6.390553,-1.617533,6.709425,0.485898],[-0.531986,8.172531,7.484439,0.428702,8.062826,5.947649,7.756915,-0.480308,0.023719,1.682280],[-9.942690,-5.606197,-7.454408,0.076777,-1.884105,-7.404776,4.081135,8.894237,-9.771189,-4.656898],[-9.477139,-2.812045,-5.750416,-0.933015,-6.316063,2.374200,7.389588,-9.130800,3.372803,8.680846],[-4.709122,-1.595950,-4.576133,-6.749743,-9.790954,-9.001352,-0.875204,2.396589,6.024882,-1.314700],[6.860473,-3.373306,2.496245,-1.937957,-7.493586,-5.892715,-1.415234,-3.349585,7.300209,7.821288],[4.843685,-8.013424,-8.854895,7.479316,9.809727,5.346565,9.282510,8.910410,-4.255032,8.546190]],[[-0.677018,5.317569,7.039872,7.204978,-3.241555,-6.899287,-4.705398,6.484896,-4.865340,-6.210087],[-8.959810,-4.258237,2.399813,-5.020616,-8.432707,5.081459,9.943040,8.930277,-3.485280,1.222785],[2.481650,1.780520,-7.663575,-3.734693,0.107754,2.811397,-2.559997,4.213634,9.565623,1.548223],[-2.270223,-8.463996,6.212931,-7.398795,-3.871197,-8.708791,0.256339,4.652063,-5.621176,5.416712],[-6.720476,5.384633,-2.584439,7.521671,-5.162433,2.685006,1.285874,-0.685088,1.481526,-1.664607],[6.649236,6.650464,-1.065933,-5.053448,-9.165954,-8.263524,-6.271669,8.099281,6.666156,8.388981],[9.261740,-4.563319,-8.226716,0.026811,1.484090,-7.123603,-8.365901,4.606728,-4.009314,5.614703],[1.179951,-0.972972,6.388683,1.525663,-3.904351,-9.598953,0.105183,-5.497487,-8.390718,-8.032615]],[[7.254869,-8.050254,-2.565201,7.690507,3.124285,-5.849246,-5.154486,1.054490,-1.341014,-5.467369],[8.193221,3.908749,2.726246,-1.505011,5.249614,4.719573,0.398952,5.825391,-7.391001,8.989147],[7.786104,-1.099953,-0.835441,-9.059588,-6.669087,4.685417,1.038975,3.046733,-5.620878,-8.019016],[-1.575280,-7.863878,0.268275,1.411258,-5.168482,-8.508830,-8.587536,-9.497914,-4.524942,-2.879686],[5.173690,-5.912185,-5.995605,-5.828359,1.955358,-6.832167,-0.582472,-8.944370,-2.475377,2.191482],[-5.109223,4.706113,2.454496,3.308127,-5.287513,-3.673923,1.208305,-8.721991,-8.256619,-8.105234],[-6.029707,8.452861,6.836436,-3.599741,-3.360784,0.969822,2.327063,6.818295,-0.796445,-2.244688],[-7.737684,-3.160923,0.640842,9.867900,2.460324,-5.667062,2.691989,9.294179,5.156681,-4.419728]],[[8.176109,-4.013846,-0.975025,2.152614,-3.341537,4.278223,9.685777,-1.062855,4.911929,-4.019694],[0.261741,-2.494113,4.815801,1.978000,2.391505,-8.067900,1.444680,-9.890291,-6.970380,-1.122912],[7.308993,-7.099955,-4.081246,-6.494645,-1.040452,-6.453101,1.974973,-6.103351,-4.899377,0.289210],[-3.183388,1.197665,-5.161924,3.563574,2.387558,3.777179,-7.146122,7.248350,-0.631154,-5.779861],[5.260686,-7.490074,3.361103,2.333316,-1.774266,4.058908,-5.258277,4.711375,-7.354458,2.971428],[-0.977116,7.263260,-4.019788,8.359477,-4.464812,-0.026216,5.173231,-0.141444,6.740393,-4.434828],[-5.335021,-0.834665,-7.502186,3.851002,8.656194,9.600302,6.923520,-0.157931,8.816701,-9.282782],[9.031556,9.165956,2.152365,5.494558,4.516701,-1.261109,8.256148,-5.043330,-5.291225,6.977473]],[[4.523221,-0.056711,5.208941,-8.584331,-9.812821,-7.776400,0.006867,-6.627663,5.140004,-3.178669],[-2.064462,-4.319154,7.069718,4.286665,-5.557769,4.718814,8.844737,-2.352439,-5.641427,-4.119171],[5.319488,-3.858801,-2.049814,0.975697,-2.933252,5.141732,-4.176993,8.450920,-0.343045,-3.141807],[0.330622,-9.873202,5.822116,6.226241,-4.431124,-7.616845,-4.641193,-5.001976,-3.363833,-9.363522],[5.543738,0.384950,7.760522,-6.450684,0.869132,-6.088031,-2.392824,-8.739624,8.714760,-2.622365],[5.114508,6.663004,7.207223,-6.307562,5.416037,9.115780,-1.966806,-8.421830,1.744018,-0.980546],[0.686447,6.628994,1.790268,4.551987,4.420700,-5.280713,2.478316,-2.662252,-4.669288,-2.821183],[2.460191,-1.212820,-4.874633,2.017147,5.418511,-4.899407,-0.893641,2.551011,4.036365,-2.031133]],[[-0.301678,1.274319,-0.893420,-6.286760,3.380774,-3.449626,-2.697573,-4.599206,1.432567,-3.338365],[4.536900,-0.075496,-3.281750,-9.511664,-0.836793,-2.583641,2.128810,-6.009074,5.138304,-9.184058],[3.823379,4.881803,-3.872444,-1.655036,5.988750,-3.997453,-9.616624,-3.991384,-3.304731,-6.800855],[9.377021,0.723723,0.248181,-3.530243,3.573311,0.652909,4.084255,-4.830932,-9.052035,-9.802474],[-3.082180,2.618700,7.989193,2.936864,-9.924593,-3.896792,2.106216,-8.431178,-9.943845,5.156342],[-2.690931,-9.132766,-0.161287,3.736382,-5.114104,-1.462306,-3.588298,3.311751,-4.820496,-4.994831],[-5.668193,-3.011447,-4.020523,-4.122312,-4.059516,-5.628427,-7.983721,-0.202326,-9.437126,-6.605272],[9.638435,8.636471,-6.043801,4.801409,7.739856,0.419641,8.613597,2.359671,-8.710461,-2.305107]],[[-7.394833,1.479984,0.792221,5.042570,2.923517,-7.612263,-6.963615,8.845696,-2.903629,0.365736],[-8.577848,-8.520495,0.671520,7.056062,1.282882,-7.461431,7.707515,1.752250,8.286114,-6.044997],[-0.321051,-2.862080,-1.244117,3.152903,-3.708187,-5.493660,3.085308,-9.605759,6.509988,-4.479143],[2.414516,-0.863826,5.643931,-2.919961,-3.049420,3.082740,7.056033,7.616239,4.558524,6.590061],[-6.377736,-0.630390,-8.317228,-5.215979,4.493662,-8.268187,3.114178,6.593468,-2.011683,3.021423],[-2.329465,4.865641,-2.165022,3.596116,1.970162,-6.403384,-3.518405,6.171394,0.286916,-7.420016],[7.341994,-2.468677,6.468982,1.393489,0.895741,-1.064718,4.271156,5.674524,1.581319,-7.966917],[7.949991,-6.968697,6.719584,-6.253954,9.765727,1.007529,-8.174972,2.627844,8.716744,6.952499]],[[3.563699,-4.995828,-1.085788,7.650774,-2.009120,-6.304323,-2.464723,-6.444691,1.168508,-0.856623],[4.234336,7.520069,4.103987,-8.190498,1.448248,4.974089,6.168346,0.411783,-3.198276,5.739330],[-3.164465,-1.999891,-2.907170,9.117307,-6.535399,-3.612519,0.770606,4.675671,-4.559450,-5.065607],[6.288124,7.274316,2.424009,-3.671604,0.063839,7.738358,-3.152002,-5.396593,-2.420646,-7.615226],[2.783471,-9.643757,-4.397926,-1.932815,7.076212,-0.744690,-5.028555,-9.004538,-4.029848,2.603967],[-4.699023,-0.068007,2.319395,-1.022833,-1.906062,-6.397102,4.635750,-4.387854,-5.918895,2.201085],[9.887271,-7.055308,2.992249,-9.363104,-7.456167,0.972606,-0.657813,7.698892,-3.166232,1.122473],[-7.498935,-9.054770,-7.444345,0.329145,1.680785,8.882078,-6.512130,2.657114,-3.657219,7.112874]],[[6.666704,5.218126,-7.859371,4.496790,-4.830748,6.646096,-6.483515,6.019731,0.090454,2.581757],[7.959449,6.715560,0.163970,5.025017,-9.426675,-0.470502,1.867820,1.802483,-1.406620,-4.036503],[1.040965,6.042604,8.843354,0.888535,7.191856,-4.514499,4.722606,-2.196032,-2.264564,-8.719165],[-1.529622,-6.287429,-1.706095,-7.034774,9.816529,-8.994023,-1.690000,-0.066413,-6.100897,-7.656053],[5.387151,-5.228308,1.607313,6.814593,-3.118346,-8.148800,-5.467402,-3.745703,5.253867,2.785120],[8.114106,8.562892,7.915922,6.176765,6.238137,0.057831,-6.115911,0.653546,-5.516568,2.369954],[-2.134675,9.716247,1.554751,-7.812772,-6.778812,5.649526,4.670613,-4.203804,-0.329304,-4.671592],[3.679170,-7.898065,2.725986,9.050572,-4.130184,-8.621589,4.528150,9.275700,2.505175,2.775133]]], dtype = "float64")#candidate|16313|(13, 8, 10)|const|float64
uop_16314 = relay.sin(const_16313.astype('float64')) # shape=(13, 8, 10)
output = relay.Tuple([uop_16314,])
output2 = relay.Tuple([uop_16314,])
func_16323 = relay.Function([], output)
mod['func_16323'] = func_16323
mod = relay.transform.InferType()(mod)
output = func_16323()
func_16324 = relay.Function([], output)
mutated_mod['func_16324'] = func_16324
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11412_call = mod.get_global_var('func_11412')
func_11413_call = mutated_mod.get_global_var('func_11413')
call_16351 = relay.TupleGetItem(func_11412_call(), 0)
call_16352 = relay.TupleGetItem(func_11413_call(), 0)
func_6366_call = mod.get_global_var('func_6366')
func_6368_call = mutated_mod.get_global_var('func_6368')
call_16353 = relay.TupleGetItem(func_6366_call(), 0)
call_16354 = relay.TupleGetItem(func_6368_call(), 0)
var_16357 = relay.var("var_16357", dtype = "float32", shape = (6, 7, 15))#candidate|16357|(6, 7, 15)|var|float32
bop_16358 = relay.bitwise_or(call_16351.astype('int16'), var_16357.astype('int16')) # shape=(6, 7, 15)
bop_16361 = relay.bitwise_or(call_16352.astype('int16'), var_16357.astype('int16')) # shape=(6, 7, 15)
output = relay.Tuple([call_16353,bop_16358,])
output2 = relay.Tuple([call_16354,bop_16361,])
func_16369 = relay.Function([var_16357,], output)
mod['func_16369'] = func_16369
mod = relay.transform.InferType()(mod)
var_16370 = relay.var("var_16370", dtype = "float32", shape = (6, 7, 15))#candidate|16370|(6, 7, 15)|var|float32
output = func_16369(var_16370)
func_16371 = relay.Function([var_16370], output)
mutated_mod['func_16371'] = func_16371
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6088_call = mod.get_global_var('func_6088')
func_6090_call = mutated_mod.get_global_var('func_6090')
call_16436 = func_6088_call()
call_16437 = func_6088_call()
func_5572_call = mod.get_global_var('func_5572')
func_5577_call = mutated_mod.get_global_var('func_5577')
var_16439 = relay.var("var_16439", dtype = "int8", shape = (528,))#candidate|16439|(528,)|var|int8
const_16440 = relay.const([3.393235,-8.108433,-1.684751,-0.048803,0.717413,0.674870,5.270224,-4.825785,-6.040613,-2.547960,-6.983207,-2.761657,-8.899072,-9.646824,-9.154311,-8.956534,-5.194024,6.810108,-6.883431,-8.561911,9.879307,8.670988,2.055733,-4.249927,-7.525127,8.574696,-7.725316,1.350389,2.115874,-4.367079,-0.768708,-2.514695,-4.060511,-0.647734,8.781972,-3.863341,-1.084767,-1.477277,2.365060,8.301164,1.902920,1.884985,1.829786,-2.949602,-2.511897,-1.902960,-3.979820,5.503017,-9.910916,7.611320,3.877148,-5.054456,-0.573496,8.959409,-2.917023,0.597815,3.164703,2.690644,-9.649655,-7.519784,9.421162,8.886247,9.063751,-2.363208,-5.992851,2.049733,5.805201,6.835154,-5.113507,-8.315780,-1.679207,2.535251,-1.286743,-1.835389,-9.720060,2.137846,6.652343,2.661730,9.148951,8.214435,-1.726646,3.864103,-7.182199,9.630550,-1.621991,9.484973,9.725939,-7.992135,-3.480093,-5.525923,8.759854,-6.688033,2.708709,6.287129,4.456883,-5.305410,-5.742017,2.121741,9.612556,6.441970,-6.294909,-2.030371,0.832225,7.640015,5.870588,7.065270,2.614721,1.060427,7.863878,-4.612164,6.673013,-3.990606,-3.115506,9.829558,-6.503207,2.686576,2.135412,2.640126,8.687007,-2.583913,-5.912598,1.180360,9.752989,-9.800465,-9.289039,1.754526,-5.061954,4.294821,-3.750449,-2.423160,9.732296,4.228028,6.863628,-2.284761,-8.016619,0.728351,-7.391971,-5.155919,6.105441,-8.456318,6.404754,-2.977195,-7.002065,1.450598,-7.284350,2.931423,9.198914,-1.707561,6.419370,7.905776,-5.764746,4.006181,-8.930493,-5.969380,-8.190366,6.010631,7.010554,7.969962,1.857693,-8.292738,-6.072997,1.547098,-4.076031,-0.242764,-5.149689,6.038658,-9.613307,2.559149,8.025277,1.015400,4.283647,5.650117,-3.602640,-6.149898,5.313889,9.867278,-6.041626,-0.928872,-6.272238,0.795997,-1.402293,6.279379,0.040389,0.745115,3.682168,-3.382938,3.682629,8.849970,8.755425,2.223910,-5.964343,-3.190509,-9.268370,1.336513,-4.524299,6.880788,-7.317148,3.571136,-3.292729,-7.869355,5.612906,-4.597762,-2.850158,-0.196281,9.134540,7.905510,-8.293751,-9.852946,-5.624438,-4.401016,-8.613427,8.998436,-1.858487,-5.917508,-7.101029,-4.106488,-4.455721,-9.539325,-8.865419,-4.004482,6.792963,7.201544,0.145533,-5.916342,-8.591002,-2.602236,-7.088780,7.191160,5.779798,1.113806,-2.048354,4.149159,-5.188973,8.136715,4.881184,-3.008771,3.720795,-5.859468,-1.611384,-1.112399,1.433392,4.359455,0.654742,3.229405,1.980305,-9.660561,-1.516183,-7.141591,2.724777,-3.387517,6.423611,3.116174,-8.408623,8.216549,-5.566246,8.946088,4.086398,-3.662804,5.891820,5.441295,-3.716783,-6.593326,4.168456,-6.583218,9.425766,3.638887,4.789276,2.228970,3.687596,-7.319352,5.935091,9.911218,7.622845,0.116786,5.598000,3.205200,-2.740906,0.623142,-1.792967,-0.707401,-2.138228,-8.255466,2.942620,2.757872,-5.877968,-7.194764,-4.342498,-7.396997,9.340038,-0.872124,-4.255252,0.283951,-9.687498,9.094896,-1.840737,-8.584211,9.859957,-7.523463,-4.792778,3.371814,-4.728100,-9.591683,-2.065886,-7.957831,-4.173357,-6.047605,-9.961094,3.760746,-8.813356,1.767437,1.499802,9.407015,-6.371391,2.241088,-1.772040,6.535209,-3.846717,7.924077,0.291242,4.980535,-3.426496,-8.857431,3.239854,2.177579,-6.727761,-0.863015,-9.073452,2.067747,1.153224,-2.699918,5.890613,-4.001508,2.621616,-0.317673,8.971433,0.964130,-5.264079,-3.190877,3.710569,0.341004,-7.438934,1.132147,-8.220263,7.307805,5.740601,7.630382,-4.600233,3.592602,7.784572,-0.087847,9.432926,-2.091023,3.479805,4.921532,7.436530,8.586166,-4.808526,-8.374572,8.884507,-5.138086,8.052309,3.576453,-8.907894,-4.476199,8.198984,-8.070017,-2.133107,-0.064668,-8.853650,-3.672410,-2.564561,7.090227,0.670022,4.601798,-5.381160,2.607365,7.705820,8.662762,0.458178,0.317736,-5.815104,4.664748,0.789508,-3.718164,7.016481,3.197194,-6.403641,-2.047876,-2.203137,-9.428035,-9.721439,2.632380,1.212221,8.283116,-4.953975,4.920718,-3.795553,5.927557,-7.454402,5.076040,-0.004369,5.555978,-1.188475,-5.976853,-8.266440,3.610000,4.487657,-4.810839,7.135571,0.373426,7.008164,-4.029985,-6.205721,9.824518,9.474141,-0.904973,1.230687,6.518829,-5.045696,-1.691995,-7.906731,0.521165,-2.910596,-0.230955,1.808262,9.505273,-0.430974,-6.755018,0.775309,2.734816,3.373280,9.598611,-4.480516,-1.859860,-4.907021,8.429327,-9.638180,-0.890214,-6.119503,-7.676963,4.831291,-6.005972,7.425946,-9.680927,-7.735800,6.473424,4.150990,-1.627502,1.267978,-2.581009,-4.747527,1.685217,-0.886804,1.665416,-7.825454,-2.941432,-2.520529,-0.655352,-0.794637,-2.223648,-0.080203,-9.420165,-0.001977,7.019599,3.095155,-9.087881,0.375120,-9.326248,1.799813,2.120424,4.522704,6.543562,-1.238301,3.798476,-8.566159,-2.646515,5.345817,7.160404,0.274646,-2.559207,-6.523923,-9.540958,-9.994685,-2.068736,8.250556,8.596904,2.679284,-4.271080,-8.888981,-1.469668,6.233041,-6.515159,2.504609,6.745556,-7.895872,2.675501,-5.159773,5.376200,-4.484079,-9.712196,9.986459,-9.448607,1.034553,-2.020213,-6.136889,-1.391760,0.108773,3.891969,-5.617925,-3.259005,-1.508393,-6.735221,2.354898,9.139757,6.605137,3.774322,7.141781,4.500359,-7.117030,2.168832,9.352458,6.166389,-4.216154,6.156326,6.028780,8.272113,-0.274774,-1.806646,1.041057,3.810921,2.074705,6.070840,6.371848,4.507074,1.059679,-0.525744,1.538989,0.415251,-7.207266,-2.342217,-9.039779,4.628482,5.912823,6.185015,8.899422,4.467881,0.699767,-3.469662,-4.758031,6.927074,-7.281103,-5.681791,6.167795,2.409449,-1.552285,6.030989,9.649141,4.381510,-3.722513,-3.340429,-7.417407,-3.326411,-3.207379,-3.500067,2.224708,-0.333923,0.924876,0.246208,5.929858,2.309827,-7.328408,-1.910835,4.201570,5.078283,-5.682517,-6.601624,3.876122,1.729896,6.531990,-4.173578,7.754041,-7.493326,5.536483,2.925496,-8.349505,8.408661,3.486932,-4.214537,1.427379,-7.775730,5.805395,-5.019243,0.156714,4.622625,5.491687,2.722183,-6.537841,-5.213148,5.947247,-8.001274,-4.885992,1.445363,-5.214450,-9.597748,6.794402,5.259227,5.585694,5.899924,1.197684,7.939205,2.025399,9.348365,-5.090559,9.660298,8.367592,9.011874,1.366978,7.329268,8.263078,-5.972163,-4.289607,5.932321,7.607355,8.323577,4.280104,-9.367849,1.387834,1.459844,2.122768,-0.727599,-5.130718,-1.323755,-3.019812,5.056908,2.262358,-4.357992,-9.884530,-3.855819,-2.383199,-2.811936,-3.680790,-9.733915,7.080952,-9.864832,5.765763,3.110789,-0.604176,9.628347,-9.752218,-6.501254,-2.675680,7.841204,8.320165,2.093113,5.675949,2.033031,-3.932455,-0.203360,6.330053,-1.045273,8.492901,5.702357,0.859945,-2.016722,3.227704,8.836860,7.006254,-0.282269,-1.470306,-2.782761,-2.825527,-7.510967,-6.127374,-9.067109,8.587640,-6.985571,-2.246529,-3.211889,-2.443655,5.036541,-6.200343,-4.289117,-3.271459,-8.787970,-3.785277,-1.788702,-3.267942,8.935754,1.476906,8.639054,-3.051613,7.794433,6.675282,-8.117882,-4.781732,9.955457,-3.797800,-0.673329,9.397609,-9.454594,-0.616218,-1.638827,5.843980,7.658980,8.035384,5.625251,-1.140892,-8.723010,-2.157814,2.890364,2.567096,2.485030,-3.027868,-8.350158,-3.962325,7.060044,9.476045,-6.801932,8.749204,-9.968165,-6.434745,-4.968135,-9.762801,-5.395482,9.425342,-1.378436,-1.727508,3.053501,4.016373,-1.608054,-2.090620,-4.756959,8.340235,-4.729655,-3.135275,4.277071,-8.035480,0.100661,2.644072,0.571307,-9.579698,5.706248,-7.509251,3.453637,-4.395044,8.378904,-7.034176,-4.261434,-6.287214,-4.885396,6.838858,2.944648,-3.735390,1.956054,-1.048040,-8.537154,-6.546008,-3.546356,-3.839239,-3.383315,4.054397,6.918278,-5.888309,-5.076314,-1.735292,8.926666,-0.766911,1.590428,-5.835265,-2.559215,-8.300288,2.052469,-3.487662,0.858806,4.904839,7.801178,1.939935,-1.034226,2.763505,1.522501,-8.231712,1.102189,8.080468,0.269379,0.114462,2.234352,-4.622447,-3.680702,-1.018445,-2.645354,-9.275191,-2.574474], dtype = "float32")#candidate|16440|(792,)|const|float32
var_16441 = relay.var("var_16441", dtype = "uint16", shape = (336,))#candidate|16441|(336,)|var|uint16
const_16442 = relay.const([9,-10,6,-2,7,-8,-4,10,-3,7,8,7,-10,3,8,-6,-7,2,6,8,7,4,7,5,-4,-5,2,6,-10,3,5,5,-6,9,-2,5,-5,3,4,-9,2,5,-8,7,-2,-6,-2,2,7,-1,-8,-10,-7,4,-1,-4,-9,-9,1,-6,10,1,4,-9,-5,-2,7,8,7,7,-5,2,-9,-4,1,-3,-4,3,8,-9,-2,-5,-4,-5,5,10,-3,5,-5,2,8,9,10,10,-3,2,-1,-6,-2,-8,3,6,-10,9,1,-1,7,-3,-1,-6,3,2,6,10,1,4,4,3,4,-9,-3,8,-10,8,5,8,3,3,-9,7,1,-7,5,10,2,-4,2,3,-2,-3,-7,6,-2,9,-2,-5,-10,-7,-4,4,-6,9,10,-10,-4,-1,6,1,4,1,-8,8,-1,-5,-2,9,7,8,3,-10,-2,8,-6,-8,-10,-8,-5,6,-1,-1,10,-3,-9,4,8,8,5,-10,6,-5,6,5,-10,-7,4,6,-7,2,-1,-7,10,-10,1,10,-7,6,-8,5,3,-9,-10,-2,-5,5,1,-6,5,-8,6,-6,2,-4,-4,-8,-1,-4,3,1,10,-5,-6,-4,7,-9,-1,-5,10,5,10,-1,-10,3,-2,-5,-10,3,-1,9,8,-2,5,-4,-9,-7,10,-10,-4,8,-9,8,3,-7,8,-4,-4,-7,-6,-3,8,3,-9,-2,8,-9,-5,8,8,-4,-1,-3,7,-5,-1,-9,2,10,-5,-6,-6,-1,2,9,-7,6,-3,7,3,4,-2,-4,5,6,10,-4,-7,-5,6,-10,-9,3,9,2,1,5,-9,6,7,-3,-5,1,-1,10,-2,-10,-9,3,-2,10,-10,-5,-6,-2,-6,-7,-3,-3,-7,-2,5,10,-5,-2,5,-10,-8,6,7,-6,5,4,-2,-2,9,-9,-6,-8,9,7,3,3,-4,5,6,-4,5,-9,-4,9,3,-10,5,1,10,-10,-7,6,1,-3,-7,8,-7,-10,-7,6,4,-10,-6,5,-9,10,-9,7,2,-6,7,10,2,6,-8,1,10,3,2,-3,2,-9,2,10,4,-2,4,6,3,-5,-6,-4,-8,4,4,3,1,-8,7,-8,8,3,-3,7,-1,-6,1,-3,7,-10,-5,9,-4,8,7,-2,5,4,-4,-3,-3,4,-10,-4,-4,-10,-7,4,5,5,-7,6,2,-1,1,3,-4,-2,4,3,-10,1,-3,-1,7,10,-2,-8,-6,-3,3,-5,-5,-3,-2,-10,-3,-10,-6,-5,-6,8,1,4,-7,8,-5,8,9,10,2,-2,1,-9,4,9,-9,9,3,9,2,-9,-10,8,10,9,-8,-5,6,3,10,-9,8,-10,1,2,-7,-10,-8,4,-6,10,-7,-4,-10,10,-8,-10,-9,9,10,-5,2,1,6,10,10,8,6,10,-9,-6,1,5,-6,2,-6,2,2,2,9,-2,3,-6,-3,4,9,1,8,2,2,10,10,-8,5,2,1,5,-5,-2,9,2,-6,10,1,-7,-5,-3,-1,9,2,-4,-7,-8,1,-3,-5,-2,10,-2,3,5,8,-3,2,7,-5,-8,1,7,5,-4,-6,9,6,8,3,-9,5,2,-6,-1,1,3,5,-3,1,9,2,3,3,9,-9,9,-10,1,-1,7,2,-3,2,-7,1,-10,6,-5,7,7,2,-1,-8,-10,-7,-10,-9,-5,-8,3,10,4,-7,7,3,-1,9,-9,4,-5,-6,-9,-9,-3,-9,-8,-10,-3,3,10,-3,3,2,-10,-9,9,-1,10,5,8,-2,10,-1,6,4,-10,-6,10,4,8,6,-1,-8,-6,-6,6,3,7,-1,-7,-10,4,9,6,6,9,1,-9,-10,-1,-3,4,3,7,-6,-8,1,-4,8,8,-4,-7,6,6,-5,4,-4], dtype = "int8")#candidate|16442|(729,)|const|int8
call_16438 = relay.TupleGetItem(func_5572_call(relay.reshape(var_16439.astype('int8'), [8, 6, 11]), relay.reshape(const_16440.astype('float32'), [792,]), relay.reshape(var_16441.astype('uint16'), [336,]), relay.reshape(const_16442.astype('int8'), [729,]), ), 3)
call_16443 = relay.TupleGetItem(func_5577_call(relay.reshape(var_16439.astype('int8'), [8, 6, 11]), relay.reshape(const_16440.astype('float32'), [792,]), relay.reshape(var_16441.astype('uint16'), [336,]), relay.reshape(const_16442.astype('int8'), [729,]), ), 3)
output = relay.Tuple([call_16436,call_16438,var_16439,const_16440,var_16441,const_16442,])
output2 = relay.Tuple([call_16437,call_16443,var_16439,const_16440,var_16441,const_16442,])
func_16445 = relay.Function([var_16439,var_16441,], output)
mod['func_16445'] = func_16445
mod = relay.transform.InferType()(mod)
mutated_mod['func_16445'] = func_16445
mutated_mod = relay.transform.InferType()(mutated_mod)
func_16445_call = mutated_mod.get_global_var('func_16445')
var_16447 = relay.var("var_16447", dtype = "int8", shape = (528,))#candidate|16447|(528,)|var|int8
var_16448 = relay.var("var_16448", dtype = "uint16", shape = (336,))#candidate|16448|(336,)|var|uint16
call_16446 = func_16445_call(var_16447,var_16448,)
output = call_16446
func_16449 = relay.Function([var_16447,var_16448,], output)
mutated_mod['func_16449'] = func_16449
mutated_mod = relay.transform.InferType()(mutated_mod)
var_16475 = relay.var("var_16475", dtype = "float32", shape = (16, 7, 6))#candidate|16475|(16, 7, 6)|var|float32
uop_16476 = relay.erf(var_16475.astype('float32')) # shape=(16, 7, 6)
output = relay.Tuple([uop_16476,])
output2 = relay.Tuple([uop_16476,])
func_16484 = relay.Function([var_16475,], output)
mod['func_16484'] = func_16484
mod = relay.transform.InferType()(mod)
mutated_mod['func_16484'] = func_16484
mutated_mod = relay.transform.InferType()(mutated_mod)
var_16485 = relay.var("var_16485", dtype = "float32", shape = (16, 7, 6))#candidate|16485|(16, 7, 6)|var|float32
func_16484_call = mutated_mod.get_global_var('func_16484')
call_16486 = func_16484_call(var_16485)
output = call_16486
func_16487 = relay.Function([var_16485], output)
mutated_mod['func_16487'] = func_16487
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10179_call = mod.get_global_var('func_10179')
func_10180_call = mutated_mod.get_global_var('func_10180')
call_16495 = func_10179_call()
call_16496 = func_10179_call()
output = call_16495
output2 = call_16496
func_16505 = relay.Function([], output)
mod['func_16505'] = func_16505
mod = relay.transform.InferType()(mod)
output = func_16505()
func_16506 = relay.Function([], output)
mutated_mod['func_16506'] = func_16506
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12189_call = mod.get_global_var('func_12189')
func_12191_call = mutated_mod.get_global_var('func_12191')
call_16531 = relay.TupleGetItem(func_12189_call(), 0)
call_16532 = relay.TupleGetItem(func_12191_call(), 0)
output = relay.Tuple([call_16531,])
output2 = relay.Tuple([call_16532,])
func_16546 = relay.Function([], output)
mod['func_16546'] = func_16546
mod = relay.transform.InferType()(mod)
output = func_16546()
func_16547 = relay.Function([], output)
mutated_mod['func_16547'] = func_16547
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13569_call = mod.get_global_var('func_13569')
func_13571_call = mutated_mod.get_global_var('func_13571')
call_16558 = relay.TupleGetItem(func_13569_call(), 0)
call_16559 = relay.TupleGetItem(func_13571_call(), 0)
func_16369_call = mod.get_global_var('func_16369')
func_16371_call = mutated_mod.get_global_var('func_16371')
const_16585 = relay.const([-5.580483,6.113945,7.113753,-6.163998,-9.949129,3.228144,-3.450318,1.297064,6.652768,1.536179,1.160727,-0.172300,4.154929,4.544430,5.704712,9.652987,-0.966785,-7.084038,3.038022,-8.779810,-5.889423,-2.971447,-3.419200,1.676616,0.782158,8.333530,-6.694337,-8.328598,6.992764,-4.899941,5.919685,-1.235589,-8.222032,4.753150,7.733119,6.662998,9.880878,-5.450759,-7.506256,7.449145,5.175734,-1.791970,-6.557205,9.725144,-8.808408,-5.716778,2.030559,7.364942,5.559719,3.182745,-9.845628,-5.089660,-3.720399,-7.193880,-7.909856,-4.205629,-4.134386,8.802203,-0.081265,-2.848826,2.507820,-8.693793,9.860811,-1.537717,-4.383940,3.948588,6.845233,-9.387940,7.193760,-9.483828,-9.145451,6.211873,9.111518,-9.755769,8.770476,-7.664006,1.000545,8.073607,-3.108007,-5.974226,7.359136,8.506000,1.296386,-6.188604,-2.585555,6.197617,2.979800,1.284666,7.120981,-4.343232,-1.205576,4.333324,2.270623,-1.547060,5.607287,-9.284211,8.190914,3.689784,2.825812,6.567496,-6.912007,0.313758,5.358437,-1.208751,-8.618429,0.010234,7.376069,-3.107741,7.547970,-3.030731,8.770003,2.306602,-0.340120,-8.644239,-4.988982,-0.180464,-1.251445,-1.358846,-9.212043,-1.866846,-6.763363,-2.100515,6.849873,-0.318131,3.157898,5.755100,-4.843252,6.552475,4.187402,-3.799437,8.197855,-2.853963,-3.533090,8.690024,6.845816,4.742141,8.908695,1.614988,0.781016,3.697470,-0.231527,3.814349,1.332504,1.642166,-1.670894,-8.597822,-6.591510,2.825272,-8.027016,-0.585842,5.051308,7.205524,9.201093,8.844473,-6.575920,7.846991,3.652577,9.730154,-9.995853,6.307599,1.543937,-9.472739,-5.376397,-5.128777,2.375635,-1.043251,4.800767,3.144312,-3.348537,1.246731,-4.626291,6.112409,7.903484,-4.644549,1.179801,1.286115,-1.534926,9.539614,4.928164,9.774645,-1.822021,9.982761,8.862712,-3.219682,2.884778,-4.545624,-2.292061,6.221820,5.639335,1.255543,2.096440,2.059070,4.145693,-5.637683,-7.774729,6.939698,9.466761,-2.424844,2.497709,-1.695357,-3.413647,-2.316204,-3.176932,0.625533,-3.351477,8.506628,5.598495,9.835550,4.662683,2.744450,5.326560,-7.590880,-4.792887,9.604242,6.635670,-5.784500,-3.047373,-2.356724,2.692475,-9.440064,0.890466,6.289729,-5.138675,8.129878,5.699114,-2.306751,5.744249,5.029352,7.995457,8.609444,8.592414,4.606425,8.644765,-8.037850,2.142479,-5.499545,-8.576819,3.321912,-0.442642,-6.707563,0.713434,-1.687457,9.217363,-9.784513,-2.815255,7.033779,-7.527358,7.540664,7.840338,-2.113546,4.964165,2.072465,-0.853163,-4.981824,0.364223,-2.657006,4.513623,2.314598,6.432258,-1.289577,1.097187,4.465640,-1.992251,0.689484,-8.581526,-4.100754,-8.223361,-3.719830,-1.639515,9.242935,2.891387,-1.374267,-0.598157,-6.331719,-2.667791,8.445191,-5.060492,6.571790,-5.506981,-6.239228,-2.264660,-2.429735,-3.100263,3.979408,2.991039,-1.038479,9.881191,-8.888282,-8.820549,6.388429,3.925434,1.269090,4.326451,-7.343641,8.472420,-3.735967,1.001698,7.725637,2.205686,-6.965375,1.573982,-9.947915,1.923139,8.515897,2.319303,-5.823337,2.705013,-0.115234,-1.588664,-5.589216,-7.415890,8.906841,7.795353,0.949305,-1.310759,-5.975408,7.417309,-3.187064,0.601050,-9.164701,-1.368785,2.784422,-2.784821,-8.376893,-8.981362,9.191776,7.521682,-7.173031,1.296732,-0.085566,5.131405,3.233630,9.161602,0.166773,6.344619,3.693240,2.509588,-9.665596,-3.781634,-1.079072,9.742349,1.649165,4.244334,-7.523613,-6.646809,-1.548744,7.801217,-8.226280,5.043361,4.943727,4.215216,-9.893752,7.129058,8.947171,4.909789,-4.702672,9.398762,7.257704,1.230201,2.209091,-7.056145,9.830260,6.253235,-0.530239,5.992057,-8.986876,-4.387858,6.018010,7.985250,-7.032969,1.795134,9.673901,-4.530699,3.928434,-6.851214,2.717107,-5.816227,-8.405963,-1.839712,-8.918324,-7.079747,2.144721,2.703775,-3.112733,9.174534,7.848131,3.239078,6.377424,6.638154,2.034456,1.845848,3.519699,8.000757,7.867834,-8.580734,-7.660208,-5.726501,-0.373472,-1.320818,-2.443858,3.535736,-7.732571,0.325984,-8.569315,-2.104719,-0.797445,8.799213,8.937090,-4.646445,-4.254969,-0.632587,6.086411,-6.140614,-7.777636,2.484632,-7.765958,-6.571058,4.830461,-7.285474,4.060813,-5.091193,-1.664929,7.472845,-8.681300,9.012304,-3.496920,0.798389,-8.538212,6.584631,7.995187,-9.742013,0.871071,-2.946834,-6.372815,9.661922,-8.889605,9.414808,2.789217,-0.372014,-4.070192,-8.209899,-5.695651,4.541890,-0.650844,8.328343,8.717738,1.326360,2.641151,8.092622,6.234610,7.890304,-4.523865,-9.272706,1.682003,-6.261944,-7.809477,-5.837306,-1.927424,-3.977838,0.059918,-7.745851,1.087024,1.028986,-2.749111,-0.329820,0.263009,6.597517,-0.777321,5.320059,-9.457429,-2.167183,2.624543,6.511047,-3.596051,-2.159222,-9.130202,-3.944933,-6.908956,-8.098433,-7.752142,-3.812629,-4.632120,-6.391590,-7.369989,-8.064151,5.728099,6.943284,1.207598,-7.059890,-6.500081,2.819417,-5.144877,-3.255497,-2.995789,6.539607,3.638587,-9.647968,5.294536,6.531914,3.624099,-6.920146,-6.191963,4.834183,6.017046,8.139165,-7.554249,-1.625639,-1.472219,5.875155,1.215189,9.677761,8.007738,-8.749933,-4.087500,-9.978916,2.166569,-0.540496,-9.036688,9.784504,-9.211889,-1.709463,-7.510513,4.782539,3.790430,1.448136,6.364973,-6.673651,-9.799451,4.023275,-8.192564,-6.805486,-2.575326,5.977042,-3.869323,-8.823265,7.311504,3.526024,7.765984,-7.508252,-6.304281,5.102913,-8.530765,-3.530640,-7.293377,-4.708231,8.228723,-8.595894,6.219254,-1.645985,5.709179,-6.249821,4.061245,-2.153745,-2.915503,-3.706220,-6.986978,-7.980466,6.748222,1.331296,-2.883091,-5.146903,1.644205,-9.593546,-2.853459,-7.039396,0.874025,-9.675227,7.068876,-6.353952,3.341520,-1.239709,-3.607992,-0.324530,1.960159,-3.583472,5.830653,9.516785,4.007341,-8.056934,-6.320531,4.721555,7.512921,3.778921,-2.497934,5.667371,-6.858390,7.935413,-2.157411,-0.904288,-4.476396,-1.993482,6.321330,-9.101997,9.010374,-4.344256,3.106752,2.310817,0.950034,7.458763,-7.963976,2.551328,-9.959114,-4.273414,-7.074588,-6.927886,7.392803,1.305851,-8.752240,1.285436,-8.047337,-3.099471,-6.133435,-8.450465,-7.724772,6.711760,-4.322926,2.366306,-2.131904,-3.279418,-6.640794,5.074898,3.099576,-7.586532,-3.777505,9.117086,0.347954,-4.417346,2.116854,9.606763,0.558393], dtype = "float32")#candidate|16585|(630,)|const|float32
call_16584 = relay.TupleGetItem(func_16369_call(relay.reshape(const_16585.astype('float32'), [6, 7, 15])), 0)
call_16586 = relay.TupleGetItem(func_16371_call(relay.reshape(const_16585.astype('float32'), [6, 7, 15])), 0)
func_14997_call = mod.get_global_var('func_14997')
func_14998_call = mutated_mod.get_global_var('func_14998')
call_16598 = func_14997_call()
call_16599 = func_14997_call()
func_15782_call = mod.get_global_var('func_15782')
func_15784_call = mutated_mod.get_global_var('func_15784')
var_16617 = relay.var("var_16617", dtype = "uint8", shape = ())#candidate|16617|()|var|uint8
call_16616 = func_15782_call(relay.reshape(var_16617.astype('uint8'), []))
call_16618 = func_15782_call(relay.reshape(var_16617.astype('uint8'), []))
output = relay.Tuple([call_16558,call_16584,const_16585,call_16598,call_16616,var_16617,])
output2 = relay.Tuple([call_16559,call_16586,const_16585,call_16599,call_16618,var_16617,])
func_16619 = relay.Function([var_16617,], output)
mod['func_16619'] = func_16619
mod = relay.transform.InferType()(mod)
mutated_mod['func_16619'] = func_16619
mutated_mod = relay.transform.InferType()(mutated_mod)
var_16620 = relay.var("var_16620", dtype = "uint8", shape = ())#candidate|16620|()|var|uint8
func_16619_call = mutated_mod.get_global_var('func_16619')
call_16621 = func_16619_call(var_16620)
output = call_16621
func_16622 = relay.Function([var_16620], output)
mutated_mod['func_16622'] = func_16622
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12992_call = mod.get_global_var('func_12992')
func_12993_call = mutated_mod.get_global_var('func_12993')
call_16629 = func_12992_call()
call_16630 = func_12992_call()
output = call_16629
output2 = call_16630
func_16631 = relay.Function([], output)
mod['func_16631'] = func_16631
mod = relay.transform.InferType()(mod)
mutated_mod['func_16631'] = func_16631
mutated_mod = relay.transform.InferType()(mutated_mod)
func_16631_call = mutated_mod.get_global_var('func_16631')
call_16632 = func_16631_call()
output = call_16632
func_16633 = relay.Function([], output)
mutated_mod['func_16633'] = func_16633
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8428_call = mod.get_global_var('func_8428')
func_8429_call = mutated_mod.get_global_var('func_8429')
call_16642 = relay.TupleGetItem(func_8428_call(), 0)
call_16643 = relay.TupleGetItem(func_8429_call(), 0)
uop_16662 = relay.acos(call_16642.astype('float32')) # shape=(8, 10, 16)
uop_16664 = relay.acos(call_16643.astype('float32')) # shape=(8, 10, 16)
uop_16670 = relay.exp(uop_16662.astype('float64')) # shape=(8, 10, 16)
uop_16672 = relay.exp(uop_16664.astype('float64')) # shape=(8, 10, 16)
output = uop_16670
output2 = uop_16672
func_16680 = relay.Function([], output)
mod['func_16680'] = func_16680
mod = relay.transform.InferType()(mod)
output = func_16680()
func_16681 = relay.Function([], output)
mutated_mod['func_16681'] = func_16681
mutated_mod = relay.transform.InferType()(mutated_mod)
func_14095_call = mod.get_global_var('func_14095')
func_14097_call = mutated_mod.get_global_var('func_14097')
call_16710 = func_14095_call()
call_16711 = func_14095_call()
output = call_16710
output2 = call_16711
func_16713 = relay.Function([], output)
mod['func_16713'] = func_16713
mod = relay.transform.InferType()(mod)
mutated_mod['func_16713'] = func_16713
mutated_mod = relay.transform.InferType()(mutated_mod)
func_16713_call = mutated_mod.get_global_var('func_16713')
call_16714 = func_16713_call()
output = call_16714
func_16715 = relay.Function([], output)
mutated_mod['func_16715'] = func_16715
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13569_call = mod.get_global_var('func_13569')
func_13571_call = mutated_mod.get_global_var('func_13571')
call_16720 = relay.TupleGetItem(func_13569_call(), 0)
call_16721 = relay.TupleGetItem(func_13571_call(), 0)
output = relay.Tuple([call_16720,])
output2 = relay.Tuple([call_16721,])
func_16727 = relay.Function([], output)
mod['func_16727'] = func_16727
mod = relay.transform.InferType()(mod)
output = func_16727()
func_16728 = relay.Function([], output)
mutated_mod['func_16728'] = func_16728
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5200_call = mod.get_global_var('func_5200')
func_5201_call = mutated_mod.get_global_var('func_5201')
call_16781 = relay.TupleGetItem(func_5200_call(), 0)
call_16782 = relay.TupleGetItem(func_5201_call(), 0)
output = call_16781
output2 = call_16782
func_16804 = relay.Function([], output)
mod['func_16804'] = func_16804
mod = relay.transform.InferType()(mod)
output = func_16804()
func_16805 = relay.Function([], output)
mutated_mod['func_16805'] = func_16805
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13138_call = mod.get_global_var('func_13138')
func_13140_call = mutated_mod.get_global_var('func_13140')
call_16836 = relay.TupleGetItem(func_13138_call(), 0)
call_16837 = relay.TupleGetItem(func_13140_call(), 0)
output = call_16836
output2 = call_16837
func_16859 = relay.Function([], output)
mod['func_16859'] = func_16859
mod = relay.transform.InferType()(mod)
mutated_mod['func_16859'] = func_16859
mutated_mod = relay.transform.InferType()(mutated_mod)
func_16859_call = mutated_mod.get_global_var('func_16859')
call_16860 = func_16859_call()
output = call_16860
func_16861 = relay.Function([], output)
mutated_mod['func_16861'] = func_16861
mutated_mod = relay.transform.InferType()(mutated_mod)
var_16921 = relay.var("var_16921", dtype = "uint64", shape = (5, 2, 7))#candidate|16921|(5, 2, 7)|var|uint64
var_16922 = relay.var("var_16922", dtype = "uint64", shape = (5, 2, 7))#candidate|16922|(5, 2, 7)|var|uint64
bop_16923 = relay.left_shift(var_16921.astype('uint64'), relay.reshape(var_16922.astype('uint64'), relay.shape_of(var_16921))) # shape=(5, 2, 7)
output = bop_16923
output2 = bop_16923
func_16946 = relay.Function([var_16921,var_16922,], output)
mod['func_16946'] = func_16946
mod = relay.transform.InferType()(mod)
mutated_mod['func_16946'] = func_16946
mutated_mod = relay.transform.InferType()(mutated_mod)
func_16946_call = mutated_mod.get_global_var('func_16946')
var_16948 = relay.var("var_16948", dtype = "uint64", shape = (5, 2, 7))#candidate|16948|(5, 2, 7)|var|uint64
var_16949 = relay.var("var_16949", dtype = "uint64", shape = (5, 2, 7))#candidate|16949|(5, 2, 7)|var|uint64
call_16947 = func_16946_call(var_16948,var_16949,)
output = call_16947
func_16950 = relay.Function([var_16948,var_16949,], output)
mutated_mod['func_16950'] = func_16950
mutated_mod = relay.transform.InferType()(mutated_mod)
func_16232_call = mod.get_global_var('func_16232')
func_16233_call = mutated_mod.get_global_var('func_16233')
call_16991 = relay.TupleGetItem(func_16232_call(), 0)
call_16992 = relay.TupleGetItem(func_16233_call(), 0)
output = relay.Tuple([call_16991,])
output2 = relay.Tuple([call_16992,])
func_16999 = relay.Function([], output)
mod['func_16999'] = func_16999
mod = relay.transform.InferType()(mod)
output = func_16999()
func_17000 = relay.Function([], output)
mutated_mod['func_17000'] = func_17000
mutated_mod = relay.transform.InferType()(mutated_mod)
func_15860_call = mod.get_global_var('func_15860')
func_15862_call = mutated_mod.get_global_var('func_15862')
call_17021 = func_15860_call()
call_17022 = func_15860_call()
func_6880_call = mod.get_global_var('func_6880')
func_6881_call = mutated_mod.get_global_var('func_6881')
call_17034 = relay.TupleGetItem(func_6880_call(), 0)
call_17035 = relay.TupleGetItem(func_6881_call(), 0)
func_5442_call = mod.get_global_var('func_5442')
func_5444_call = mutated_mod.get_global_var('func_5444')
call_17038 = func_5442_call()
call_17039 = func_5442_call()
output = relay.Tuple([call_17021,call_17034,call_17038,])
output2 = relay.Tuple([call_17022,call_17035,call_17039,])
func_17042 = relay.Function([], output)
mod['func_17042'] = func_17042
mod = relay.transform.InferType()(mod)
output = func_17042()
func_17043 = relay.Function([], output)
mutated_mod['func_17043'] = func_17043
mutated_mod = relay.transform.InferType()(mutated_mod)
func_14063_call = mod.get_global_var('func_14063')
func_14064_call = mutated_mod.get_global_var('func_14064')
call_17044 = relay.TupleGetItem(func_14063_call(), 0)
call_17045 = relay.TupleGetItem(func_14064_call(), 0)
uop_17046 = relay.atan(call_17044.astype('float64')) # shape=(945,)
uop_17048 = relay.atan(call_17045.astype('float64')) # shape=(945,)
func_9627_call = mod.get_global_var('func_9627')
func_9629_call = mutated_mod.get_global_var('func_9629')
call_17058 = func_9627_call()
call_17059 = func_9627_call()
output = relay.Tuple([uop_17046,call_17058,])
output2 = relay.Tuple([uop_17048,call_17059,])
func_17060 = relay.Function([], output)
mod['func_17060'] = func_17060
mod = relay.transform.InferType()(mod)
output = func_17060()
func_17061 = relay.Function([], output)
mutated_mod['func_17061'] = func_17061
mutated_mod = relay.transform.InferType()(mutated_mod)
var_17086 = relay.var("var_17086", dtype = "int16", shape = (3, 1, 7))#candidate|17086|(3, 1, 7)|var|int16
var_17087 = relay.var("var_17087", dtype = "int16", shape = (3, 14, 7))#candidate|17087|(3, 14, 7)|var|int16
bop_17088 = relay.less_equal(var_17086.astype('bool'), var_17087.astype('bool')) # shape=(3, 14, 7)
output = relay.Tuple([bop_17088,])
output2 = relay.Tuple([bop_17088,])
func_17092 = relay.Function([var_17086,var_17087,], output)
mod['func_17092'] = func_17092
mod = relay.transform.InferType()(mod)
var_17093 = relay.var("var_17093", dtype = "int16", shape = (3, 1, 7))#candidate|17093|(3, 1, 7)|var|int16
var_17094 = relay.var("var_17094", dtype = "int16", shape = (3, 14, 7))#candidate|17094|(3, 14, 7)|var|int16
output = func_17092(var_17093,var_17094,)
func_17095 = relay.Function([var_17093,var_17094,], output)
mutated_mod['func_17095'] = func_17095
mutated_mod = relay.transform.InferType()(mutated_mod)
func_14265_call = mod.get_global_var('func_14265')
func_14266_call = mutated_mod.get_global_var('func_14266')
call_17114 = relay.TupleGetItem(func_14265_call(), 0)
call_17115 = relay.TupleGetItem(func_14266_call(), 0)
output = relay.Tuple([call_17114,])
output2 = relay.Tuple([call_17115,])
func_17130 = relay.Function([], output)
mod['func_17130'] = func_17130
mod = relay.transform.InferType()(mod)
mutated_mod['func_17130'] = func_17130
mutated_mod = relay.transform.InferType()(mutated_mod)
func_17130_call = mutated_mod.get_global_var('func_17130')
call_17131 = func_17130_call()
output = call_17131
func_17132 = relay.Function([], output)
mutated_mod['func_17132'] = func_17132
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12507_call = mod.get_global_var('func_12507')
func_12508_call = mutated_mod.get_global_var('func_12508')
call_17136 = func_12507_call()
call_17137 = func_12507_call()
func_14223_call = mod.get_global_var('func_14223')
func_14225_call = mutated_mod.get_global_var('func_14225')
call_17145 = relay.TupleGetItem(func_14223_call(), 0)
call_17146 = relay.TupleGetItem(func_14225_call(), 0)
func_16484_call = mod.get_global_var('func_16484')
func_16487_call = mutated_mod.get_global_var('func_16487')
const_17156 = relay.const([-1.610211,2.140284,-7.979388,2.786283,-0.851851,-6.341184,7.245078,0.001327,-1.591662,-4.003947,-5.863411,-1.145307,3.155931,5.037620,-0.285478,7.193271,-2.138804,-6.843875,3.631494,-5.223650,1.898161,-2.369783,-2.211904,-6.617546,-6.653454,-0.853218,5.383343,4.058638,-2.265014,1.589138,-7.861355,8.909687,1.877115,6.646753,1.301711,-1.250295,8.254121,9.283808,-4.036179,-3.703290,-8.392164,-5.402196,6.856831,7.618886,-4.844583,-5.144118,8.004240,5.392213,4.919025,3.189466,-8.369037,8.073563,-2.305659,7.673958,-0.691483,5.578426,3.649911,-9.593639,9.514472,-1.941818,-1.795938,1.448441,8.107581,1.238555,1.104266,3.658680,-8.730776,4.316381,-5.354053,-7.323755,-9.075575,2.505203,3.607667,-7.501073,8.681529,-4.732160,8.927227,4.873043,-4.104627,2.896007,8.264593,1.741344,8.241822,-0.974263,-9.486243,2.560522,6.960998,-1.964823,-4.413482,0.615932,2.583460,-3.148852,6.797069,0.561500,-9.826845,-3.172533,-3.841146,2.066057,-7.918609,4.607782,-6.967578,-0.005846,0.238957,-2.385055,1.695569,-7.548593,1.267647,-7.208031,-4.464253,-8.088835,-6.500076,4.794205,1.129973,2.051971,-9.830411,-4.470044,2.811075,-0.998699,3.766955,-1.450663,3.191652,8.257707,4.889711,3.971184,-0.763322,-4.298943,-2.737967,5.895910,6.611307,-2.093199,9.336827,-0.012092,9.771125,2.772274,-3.319863,3.105539,-9.594382,-9.004308,-0.468307,-7.974845,-0.144225,-2.198433,-7.972603,-7.774953,-2.947614,4.638583,6.909625,-3.808640,0.799491,5.215202,0.250114,7.815716,0.895035,-4.842338,4.682590,-4.678240,6.796269,9.490986,-4.620577,3.289748,-7.153918,-4.570932,7.037762,-3.367523,5.226232,-6.876344,-3.989178,-5.163838,3.873187,-2.719428,3.058894,-0.448710,3.069325,-9.038280,7.778000,-2.763022,4.958398,6.737340,2.334768,2.007357,2.645904,4.449580,2.839188,-3.911268,-8.025086,2.865751,0.039743,4.107199,6.046880,1.819462,4.953890,-8.564357,-1.414211,8.782120,5.896227,-6.819624,-9.967064,0.521092,-6.564313,0.961439,-8.955448,-1.664868,-3.086425,-2.385307,-2.388910,9.243627,8.835778,-4.174420,2.556288,0.924459,-1.194823,-3.963848,0.981947,-8.294107,2.532547,4.291261,-1.192288,0.223031,2.448736,5.895982,7.238349,5.064874,-1.447635,1.470775,-0.975851,3.550616,-0.096383,0.537716,3.256106,6.926178,1.387783,-5.506052,7.004826,0.241855,0.887310,-5.060266,5.805057,7.370987,3.989486,6.972445,9.574161,-8.994342,-1.175563,5.509959,5.227484,-5.519688,3.288597,-3.109990,1.894702,9.861191,-1.976089,0.043094,-8.481536,-9.232938,6.106748,0.086035,-4.823019,4.956482,2.652257,-5.444960,4.242642,-8.355058,8.313670,4.565572,-4.685485,6.292908,-1.121098,-3.095380,0.862353,-2.103028,-8.968496,7.918703,2.760488,-1.192470,8.867426,5.787718,6.372995,-2.576939,2.545444,5.307349,-1.313546,-0.970379,-6.688989,4.935055,1.325969,-9.693472,4.993794,-0.560780,-2.860161,-0.211721,4.770045,2.065232,6.162584,4.818911,8.044255,-3.189279,5.713265,5.949110,6.487338,9.266680,-6.412510,-1.786459,3.118308,6.229436,5.789502,-3.660350,2.299117,-0.817052,7.848449,4.565459,-0.879568,-2.480384,3.000295,9.704892,7.321999,2.620885,9.147913,2.226835,9.782575,-5.451013,-9.060880,8.978498,-7.553371,-7.915602,-9.140757,3.713817,5.193719,-9.809869,4.381703,8.069588,7.154130,-4.337915,1.321623,8.767166,-0.966668,9.765056,-6.209893,9.370752,-3.612558,-0.419610,5.763015,-5.233643,-8.360018,1.648075,-0.215354,-5.052658,-4.757775,-4.160950,-7.142456,-2.076804,-3.749977,-2.757508,-1.925577,9.960586,5.564660,7.642666,2.000095,-0.385823,4.922890,8.649273,-2.607682,5.422503,9.642388,-0.949802,1.125341,-9.655437,3.866023,3.125387,-1.395533,-6.414520,-4.787736,9.445716,4.366031,8.170841,8.556317,0.619029,-3.513098,-6.461592,-0.747751,-0.679447,3.425457,3.679095,7.124581,2.957143,-8.466087,2.983101,7.225613,-4.107474,-5.090724,1.366736,6.293688,-2.975170,2.585792,0.341699,6.608267,-1.515782,-3.974376,8.641233,-5.186072,-7.011489,8.412767,7.085975,-2.191194,3.057296,-5.985333,-5.053675,6.739437,3.948687,-6.692460,3.392427,-6.281930,-2.323853,2.824119,-8.722285,-2.549280,-4.453306,3.641430,7.063005,-6.708100,-0.326547,3.396739,-5.948993,8.077281,-5.120349,-9.400344,-8.146322,4.145561,4.099064,-1.820971,-7.002837,-7.761396,-3.533700,-9.332771,1.680072,6.907676,3.116161,9.678782,8.703814,-1.958199,-4.812258,5.014562,7.006955,3.260694,-4.022179,2.989453,4.605845,6.373275,7.787683,4.116574,-6.866389,-0.314756,5.472836,9.987957,0.487975,-7.336758,-3.725375,-6.207956,-6.954554,6.247197,2.583145,6.687549,-0.445384,0.565220,3.282489,0.398505,8.070791,0.935754,4.528768,7.992173,-8.468000,-1.803507,-4.286454,8.924286,-5.995218,7.843546,2.281531,-0.450382,6.979531,6.712565,-2.214506,0.083643,-1.006411,2.139429,7.481289,-7.187936,-1.096930,0.031293,6.908918,-0.833715,6.246422,-2.525594,-4.811094,-7.145633,-3.553685,-3.631404,-3.868269,-8.640929,7.511328,-8.066157,-9.347284,-0.023027,0.910475,4.494850,-7.815021,7.314198,6.093918,-5.709462,-8.712990,-6.571707,-1.141376,-4.888224,-8.335457,-1.498837,-1.002677,-2.180399,4.312374,5.854677,-4.544810,-5.982878,-0.300830,-5.944295,1.965204,-1.803806,-9.733332,-3.956967,8.862953,1.083104,3.443083,3.510987,0.339861,-4.685946,0.201171,0.733937,8.747935,-9.048080,3.743758,7.289650,-0.788116,6.551376,-6.165061,9.563585,-8.492226,1.699111,-1.424893,6.631332,1.814102,3.473992,7.757816,-7.722107,-3.302511,3.603311,-5.757719,-8.762395,-8.596186,4.154441,6.066026,-0.441889,6.388532,5.677289,8.216258,2.710799,-9.883532,8.848118,-3.741627,-4.349218,-7.933790,-3.030583,9.090197,7.666133,-8.089595,-4.455209,-9.637746,2.324016,7.721647,6.623543,6.557892,6.989785,4.200621,-0.973993,-0.956143,-0.030924,-8.669867,3.025941,5.791710,7.948362,6.474093,2.247963,-0.601978,-4.685075,2.195509,-5.670627,-7.044162,7.152647,-3.321552,4.434019,-4.331288,-3.998256,2.924879,-7.707317,-5.045654,-8.157609,-1.505168,7.421978,3.026509,-9.879824,-9.677455,3.360685,-8.359339,-2.516264,-2.348341,0.695549,-3.028371,0.657435,-1.552559,-5.250264,8.090072,-0.178372,5.156717,8.264576,4.096182,5.189266,-2.581485,4.585718,4.181674,-4.195637,0.945975,9.971289,2.660490,5.120912,-7.365120,-0.695457,1.803938,7.759854,8.270307,9.232386,-8.980251,-3.164356,5.266349,9.813819,1.802960,2.384228,-7.118738,-2.243305,2.225189,6.193717,-8.144634,-7.772519,-8.873620,-0.705625,-1.837184,1.551717,-8.745132,0.118545,5.850772,3.125267,-7.852685,4.931414,7.184372,-5.197313,-5.178087,-2.770197,-5.399557,-3.791313,5.893565,-4.528317,-9.211231,-0.432164,2.662868,-2.686504,4.859826,5.808455,-3.756781], dtype = "float32")#candidate|17156|(672,)|const|float32
call_17155 = relay.TupleGetItem(func_16484_call(relay.reshape(const_17156.astype('float32'), [16, 7, 6])), 0)
call_17157 = relay.TupleGetItem(func_16487_call(relay.reshape(const_17156.astype('float32'), [16, 7, 6])), 0)
func_15047_call = mod.get_global_var('func_15047')
func_15049_call = mutated_mod.get_global_var('func_15049')
call_17158 = relay.TupleGetItem(func_15047_call(), 4)
call_17159 = relay.TupleGetItem(func_15049_call(), 4)
output = relay.Tuple([call_17136,call_17145,call_17155,const_17156,call_17158,])
output2 = relay.Tuple([call_17137,call_17146,call_17157,const_17156,call_17159,])
func_17163 = relay.Function([], output)
mod['func_17163'] = func_17163
mod = relay.transform.InferType()(mod)
mutated_mod['func_17163'] = func_17163
mutated_mod = relay.transform.InferType()(mutated_mod)
func_17163_call = mutated_mod.get_global_var('func_17163')
call_17164 = func_17163_call()
output = call_17164
func_17165 = relay.Function([], output)
mutated_mod['func_17165'] = func_17165
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10383_call = mod.get_global_var('func_10383')
func_10384_call = mutated_mod.get_global_var('func_10384')
call_17187 = relay.TupleGetItem(func_10383_call(), 0)
call_17188 = relay.TupleGetItem(func_10384_call(), 0)
func_16114_call = mod.get_global_var('func_16114')
func_16117_call = mutated_mod.get_global_var('func_16117')
const_17199 = relay.const([True,True,False,True,True,False,False,True,False,False,True,False,True,True,False,False,False,True,False,True,True,False,True,False,False,True,True,False,False,False,True,True,True,True,False,False,True,False,True,True,True,False,False,False,True,False,False,True,True,False,False,True,False,True,True,True,False,False,True,True,False,False,False,True,True,True,True,True,True,False,False,False,False,False,False,False,False,True,True,False,True,False,True,False,True,False,True,False,False,False,True,True,True,True,False,False,False,False,True,False,False,False,False,True,False,True,True,True,False,False,True,True,False,False,True,False,False,False,False,False,True,True,True,False,False,False,False,True,False,False,True,False,False,True,True,False,False,False,True,False,True,False,False,True,False,True,True,False,False,True,True,True,False,False,True,True,False,True,False,False,True,True,True,True,False,False,False,True,True,True,True,False,False,True,True,False,True,True,True,True,False,False,False,True,False,True,False,True,False,False,True,False,True,False,True,True,True,True,False,False,True,False,False,True,True,True,True,False,False,False,True,True,False,True,False,True,False,False,False,True,False,True,True,True,False,False,False,True,False,True,True,True,True,True,False,False,False,True,True,True,False,False,False,False,False,False,True,False,False,True,True,False,True,False,True,True,True,True,False,True,False,True,False,True,False,False,False,False,True,True,True,True,True,True,False,True,True,True,False,True,True,True,True,False,False,False,True,True,False,True,False,False,False,False,False,False,False,False,False,True,True,False,True,False,True,True,False,True,False,False,False,True,True,False,True,True,False,False,False,False,True,False,True,True,False,True,True,False,True,False,False,False,False,True,False,True,False,False,False,False,False,True,False,False,True,True,True,True,False,False,False,True,False,True,True,True,False,False,True,True,False,True,True,False,False,True,False,False,False,False,True,False,True,True,False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,False,False,False,True,False,True,False,False,True,False,True,False,True,False,True,True,True,False,True,True,False,True,True,False,False,False,True,True,True,True,False,True,True,True,False,True,False,False,False,True,True,True,False,False,True,True,True,False,True,True,True,False,False,False,False,False,False,False,False,True,True,False,False,False,True,False,True,True,True,True,True,False,False,True,False,False,False,True,False,False,False,False,False,True,False,False,True,True,False,True,False,True,False,True,True,True,True,True,True,False,False,False,False,True,True,False,True,True,False,True,True,False,True,True,True,False,False,True,True,False,False,True,False,False,False,True,False,False,False,True,False,True,True,True,False,False,True,True,True,True,False,True,False,False,False,True,False,False,False,True,True,True,False,True,True,True,False,True,True,True,False,True,False,True,False,False,True,False,True,True,True,True,False,True,True,False,True,True,True,True,False,False,False,True,True,False,False,True,True,True,False,False,False,False,True,True,False,True,True,True,False,False,True,False,False,False,False,True,False,True,False,False,True,True,True,False,True,True,False,False,False,False,False,True,True,False,False,True,True,True,True,True,True,False,True,True,False,True,False,False,True,True,True,False,False,False,True,True,False,True,True,False,True,False,True,False,False,True,False,True,False,True,False,False,True,True,True,True,False,False,False,True,True,True,False,True,True,True,False,True,False,True,False,True,True,True,False,True,True,False,False,True,True,False,True,False,True,True,True,False,True,True,True,False,False,True,True,False,False,True,True,False,False,True,False,False,True,False,True,False,False,False,True,True,False,True,True,False,False,False,False,True,False,True,True,False,True,False,False,True,True,False,False,True,True,False,True,False,False,True,False,False,True,False,True,True,True,False,True,True,False,True,False,False,True,True,True,False,True,True,True,False,True,True,True,True,True,True,True,True,False,False,True,True,False,False,False,True,False,False,False,True,True,False,True,False,True,False,False,False,False,False,False,True,True,True,False,True,False,False,False,False,False,False,True,False,False,False,True,True,False,True,False,True,True,True,False,False,False,False,False,False,False,False,True,False,True,False,True,False,False,True,False,False,True,False,False,False,False,True,True,False,False,True,True,True,False,True,True,False,False,True,False,False,True,False,False,False,False,True,False,False,True,True,False,False,True,False,False,True,True,True,True,True,False,False,False,True,False,True,True,False,False,False,False,True,False,True,True,False,False,False,False,True,True,False,False,False,False,False,True,True,False,False,False,False,False,False,True,False,True,True,False,True,True,True,False,True,False,True,True,False,True,True,False,False,True,False,False,True,False,True,False,False,True,False,True,False,False,False,False,True,True,True,False], dtype = "bool")#candidate|17199|(945,)|const|bool
const_17200 = relay.const([7.594602,-9.962238,9.722881,-8.476728,5.982301,5.300529,-9.841568,-1.122807,4.260860,-5.085169,7.886645,9.050260,7.870041,-2.104764,-9.384365,-8.113945,-2.695095,-7.206076,-5.215975,-1.480657,8.327268,-2.040231,-1.422530,-9.075425,-9.782720,7.079842,3.196590,0.151986,5.115462,-4.082140,1.954925,8.406460,0.108434,7.182936,3.041451,-6.739869,-0.096281,-4.242913,5.818590,-1.051996,2.218177,-5.383981,5.458615,5.238922,6.921257,-1.104171,5.249407,5.895197,2.146389,-6.649477,-5.195771,-8.889940,5.097439,-0.053530,1.773424,-9.541546,-7.135605,5.635523,0.376528,9.970493,-0.547692,5.302547,0.990723,7.878752,-8.897331,4.487491,-3.683762,0.373680,-4.331609,-1.879222,-3.020027,8.065449,0.679199,-8.591396,0.868043,-1.072368,0.799226,2.026504,-5.632733,-7.566916,-9.717882,6.515169,4.026472,-4.595366,-3.774224,8.609951,-5.664561,-1.732557,7.125929,-9.359157,-5.221786,9.858761,-1.642551,-9.509928,7.956833,7.693550,1.918804,-2.109324,2.762395,-4.829704,3.150545,9.335522,-6.985149,-0.488291,-5.640061,0.562739,9.673565,5.962996,1.675878,-2.790579,-5.490680,6.651813,-2.254294,-7.844856,-9.118016,-0.338279,-3.257657,9.471436,4.506350,-5.246879,-3.780810,-0.012567,-2.338723,-4.287244,0.579081,-7.902841,-4.333266,0.808378,9.825926,9.901617,8.673611,-4.839218,5.993565,-3.213029,4.685121,-0.657773,-0.190641,9.674910,-5.134677,8.115197,-9.170375,9.936251,8.198566,0.390193,-7.114311,-3.275772,-7.665104,0.364373,5.276433,-2.182382,4.572609,-1.956401,0.750909,-5.872060,4.852696,-8.469047,-7.423478,5.519308,-6.859325,-3.605848,7.770026,-2.952312,3.160171,1.488136,-0.623598,5.889253,-3.012124,7.569259,4.869152,-6.543596,9.704489,-9.465023,-2.961986,7.622823,0.362816,-9.547744,8.487945,-4.260575,-7.864926,0.609327,-6.570431,-9.692783,-5.409213,-9.218951,-2.464170,4.753106,5.247846,9.472817,7.647922,4.711265,-0.668049,1.799688,-9.081705,0.526691,0.811505,0.046112,7.096920,-6.055828,-3.437840,-9.253605,6.775702,-2.782779,4.117703,6.039226,0.313028,8.234824,-2.287032,8.824513,1.926094,7.083899,0.149924,-4.360170,2.245667,-0.972204,3.205418,-5.349852,-3.711297,-2.216882,-2.945260,-4.773966,-2.300288,8.655886,-8.010280,5.424649,3.420133,-0.973194,-1.925881,-9.335231,-6.145089,8.936636,1.100510,-9.842640,-6.344379,0.659697,-9.485187,-3.934930,-5.159799,6.359006,-3.509907,-1.808577,-7.107581,-3.036135,8.075266,-4.163037,-7.289231,9.153145,5.629765,4.707682,-5.488050,-4.225509,5.941798,6.413458,-5.164241,-5.122024,-3.441765,0.756653,-7.593183,7.150257,-7.940358,1.521211,4.554025,-0.463679,-9.953355,-3.973812,-3.119702,-4.976365,-0.052421,4.187212,-2.029313,7.130155,-1.637697,1.972708,6.917805,-0.257127,-7.551208,-6.795698,5.934751,-8.287010,-7.609693,-1.509424,1.466250,-4.591262,3.637447,2.598582,-7.603525,-7.795825,9.137138,0.443045,-4.561955,-5.892090,9.367686,-3.338751,0.470047,-8.170574,3.290400,-3.989188,7.677271,-9.516220,2.185205,5.559110,1.493451,-3.017842,2.394968,5.032468,-7.561936,4.162941,-1.854713,-8.259874,-1.731838,0.557200,1.759722,9.903384,1.027036,1.914144,7.236730,-8.402728,-3.853607,1.722540,-0.694538,3.835255,-9.841010,-7.032377,-2.077039,-0.628657,9.679920,2.007746,8.094041,-4.966040,-2.515587,9.877536,-8.115258,-0.375925,7.056763,-5.888216,-3.532254,-4.213846,-8.880907,5.134911,1.479814,-0.719371,-2.097914,0.144502,-4.755103,-6.946017,-8.868535,-1.912736,1.109118,4.571325,-0.871730,-6.558869,-6.653383,1.288014,9.825714,6.692917,-9.829107,2.338243,3.828162,-7.897652,-7.179459,9.819754,4.996861,0.116187,-5.975808,7.092755,5.591393,-2.882055,-6.651510,-5.318338,-3.385917,-9.965951,9.058018,7.773059,-8.647452,-4.023184,-4.179506,2.859649,4.066249,-3.104051,-6.342870,-4.487344,8.867555,7.439810,-6.419052,0.788644,7.808834,0.294548,-2.492608,-5.451442,0.278107,4.012636,-6.600717,3.235191,-6.028763,6.886140,1.512326,-7.431283,-4.681611,-0.225076,4.015861,-6.336706,7.411716,-6.362153,-2.714359,-1.007124,7.249737,-1.211068,0.639139,-1.762252,8.851156,-1.938859,8.431710,-2.942021,-4.240619,4.328823,-0.326496,-3.377185,3.702539,1.285930,5.893118,1.069164,-8.362825,-9.945493,7.453146,8.376897,0.960468,3.521540,-4.946966,-9.547002,-3.268340,-3.217971,9.236080,-9.549305,8.346749,0.370591,-2.957989,-0.995640,-6.806804,-5.162385,3.222636,-8.221868,3.380868,5.516410,0.253794,-6.834111,6.175456,8.524319,2.594040,-1.687741,9.969588,-7.566210,-1.768317,-4.679201,9.460805,2.931500,-6.195261,5.835584,3.280709,6.071569,7.530882,2.620808,9.842597,4.577344,-1.509454,5.902571,7.644692,7.883017,-9.674735,-7.342692,-0.370603,-7.612235,-6.684104,-0.636599,3.958666,-3.066860,1.772512,3.672671,-1.123573,-9.943271,-3.335355,6.013914,8.093018,-7.305104,-3.515565,2.173809,-5.020573,-9.000443,-0.277935,-3.932677,-4.819246,-6.085628,-8.309126,-3.563012,-2.011384,-4.184983,9.459706,0.816104,-6.298289,-1.592332,-8.252578,-4.448476,-0.399115,8.500087,9.984400,-2.879843,0.519671,-6.497482,-1.443707,-4.370518,0.536048,-2.441213,-0.219143,1.705209,0.169090,-1.638478,0.600624,-6.766826,7.949954,-5.400190,7.356788,-3.916938,7.207027,1.956479,-6.180371,4.843758,-7.450884,-3.566906,0.730356,3.579359,0.806736,-4.157588,-8.535157,-0.447475,4.290173,4.951818,6.734815,6.815752,-7.290454,0.322431,-4.562063,-4.760976,-1.858788,-0.171293,-5.804125,-4.145699,8.295373,-7.908019,9.369737,8.903932,-3.464638,5.565259,-4.643441,-9.692774,7.388116,8.684465,-3.041422,-4.488641,-1.444673,-0.779259,3.494877,3.989251,-5.481086,-6.875386,-3.241114,-6.740622,-3.007217,7.336922,-1.425958,3.061906,6.768393,6.992260,-8.316334,-7.806708,-4.192221,6.985641,7.624819,-4.870271,4.909617,0.530113,-2.785190,-7.850625,4.749781,-9.236593,3.953600,-2.897000,-5.034117,3.322580,1.402170,-7.486605,-4.382439,7.619337,-7.812338,4.463777,2.831635,0.288299,-1.202710,-2.200449,8.798347,0.640579,3.978437,1.063371,-0.009049,-1.018624,-9.130013,8.283192,0.024007,-1.572803,2.933763,9.114760,-3.567667,-9.825956,-6.592278,2.862643,2.681925,-5.127322,-3.119428,5.268057,-2.402084,-8.725066,-9.257540,-6.850116,2.211933,2.914091,-1.389288,3.711665,3.529296,-0.845606,-9.112937,-6.173948,-3.272999,-9.929211,-4.329287,7.750035,-8.619524,-3.189441,-6.743472,-0.140352,8.222720,-5.088110,-8.530351,5.345427,7.568868,5.905152,4.061111,-7.883325,-0.205531,-6.516501,-2.865448,-4.376675,9.171009,1.156557,9.844140,0.895814,6.195014,-3.905027,0.570633,9.809991,2.805128,9.768791,-0.040071,9.556715,-4.632913,-1.900741,9.058073,3.939908,0.338370,-2.676243,-1.200264,2.931030,6.745967,2.005011,0.609990,-2.985759,1.996779,5.157047,0.372208,3.257025,-6.015714,-1.521584,-9.677083,6.132641,-7.787750,-2.801807,0.798499,-1.957349,9.518566,-6.401021,-2.098026,-2.972587,-6.105495,-8.055549,-0.129710,8.723182,8.282781,-5.301323,-8.434194,1.792917,-5.945098,7.794966,1.853085,-3.055972,-8.089083,-1.272764,3.117915,-0.323871,-8.482443,3.151566,-8.749450,-5.814777,-6.159818,0.385414,-3.583281,6.663556,-1.988951,-7.262609,-6.355655,7.511096,9.434288,-8.452597,-0.219028,-0.725040,1.199122,8.362656,6.830879,4.389857,-0.722597,-8.362454,0.211330,5.473111,-6.583265,1.586798,0.217242,7.731689,-4.248534,3.990009,7.350034,-5.035205,-2.419976,-2.601295,8.873123,7.539038,8.946456,-3.321476,-8.565081,-0.587167,7.961061,9.039053,-2.795138,-1.443208,-4.964426,6.195238,3.888638,-8.004406,9.612751,4.775540,-9.230735,6.467394,-0.838005,-4.568770,7.632760,-2.024428,5.620994,5.275691,-5.688403,-7.974649,-8.502082,-3.974801,9.912393,6.423550,-9.324605,4.037903,6.077673,-1.623430,-7.890051,-9.939657,7.235041,2.751623,0.616814,1.298210,-7.691087,-7.637638,4.149713,-9.443757,-9.038528,-4.746772,8.911524,0.482491,-5.021073,-3.581152,5.709977,-7.083314,3.204724,8.596760], dtype = "float32")#candidate|17200|(792,)|const|float32
call_17198 = relay.TupleGetItem(func_16114_call(relay.reshape(const_17199.astype('bool'), [945,]), relay.reshape(const_17200.astype('float32'), [6, 132]), ), 5)
call_17201 = relay.TupleGetItem(func_16117_call(relay.reshape(const_17199.astype('bool'), [945,]), relay.reshape(const_17200.astype('float32'), [6, 132]), ), 5)
func_12640_call = mod.get_global_var('func_12640')
func_12641_call = mutated_mod.get_global_var('func_12641')
call_17206 = relay.TupleGetItem(func_12640_call(), 1)
call_17207 = relay.TupleGetItem(func_12641_call(), 1)
func_5572_call = mod.get_global_var('func_5572')
func_5577_call = mutated_mod.get_global_var('func_5577')
const_17219 = relay.const([[7,4,-3,1,8,-2,-10,-1,8,-2,-9,4,-4,-4,-9,5,4,-1,-2,-10,6,-3],[4,-9,-10,-7,8,5,-6,-8,-7,-2,4,-7,6,7,1,8,3,-8,-8,4,-10,-9],[-9,-3,2,7,-9,-2,-4,-6,-10,2,4,8,9,-10,-10,5,-8,8,-6,-5,1,-1],[-10,-4,3,-4,8,-9,-4,7,-9,-3,-3,-8,6,-3,3,-6,2,3,-5,-2,-9,-1],[-1,-6,1,9,2,2,-1,7,-10,6,10,-5,8,-8,-10,1,7,2,-9,-9,-10,-9],[5,-6,-8,7,4,4,1,1,3,1,-9,6,-7,7,8,-10,-1,-8,-3,-8,8,10],[-6,-1,3,8,8,-8,-1,-2,-10,-1,-3,9,-5,4,6,5,5,10,-9,-7,-2,-1],[9,-1,-8,-9,3,-4,2,-4,-1,10,-4,-3,-9,-7,5,-4,-1,-3,-7,-9,-10,-1],[-2,1,-3,-9,-3,7,5,1,-3,-4,9,-5,-2,9,7,4,-4,-3,6,-1,8,-1],[5,1,-8,-4,6,5,4,-10,3,-2,-7,-1,9,-3,3,8,-5,-8,7,2,-3,-9],[-3,8,3,1,5,4,-8,-2,3,1,-10,3,-2,9,3,10,9,10,-8,-3,-6,8],[5,7,10,6,-6,7,3,3,8,-2,3,-8,-4,5,-7,6,-3,-10,8,4,4,-3],[-9,-4,2,-3,-9,-10,-3,3,-10,-4,1,-7,-9,-4,-2,1,-7,-8,3,6,4,5],[10,9,-8,-2,-6,-7,-10,9,1,-3,9,9,-5,-3,10,-6,5,-2,-8,-3,5,-1],[-2,7,2,-4,-1,2,8,5,-3,-10,-6,-3,-9,1,6,-9,4,-6,5,-9,2,10],[-7,10,6,10,4,-8,-2,9,3,-8,3,7,-3,4,-8,-7,-1,-6,-7,-9,-2,6],[-8,-1,2,-9,-3,-2,-3,-6,-1,7,-10,-9,9,-9,-8,4,1,-8,5,7,6,-6],[6,-9,7,-10,-8,7,10,-8,-3,4,-8,-7,2,10,-3,-6,-10,-9,2,5,10,-9],[8,-9,-2,4,-9,-2,5,8,10,-9,2,9,-9,6,-6,10,-4,6,-5,-7,-3,5],[1,-8,-9,3,-6,5,-7,-5,-8,9,-5,10,-9,-6,8,-10,5,1,-8,-7,-7,10],[-5,-8,-9,7,-6,-5,7,-7,3,3,-7,3,-9,10,5,9,-4,-5,6,8,-10,-6],[6,-10,-1,8,9,3,-4,-3,2,10,-1,-8,6,10,9,-5,-7,4,9,-8,-5,-8],[6,7,5,4,2,-6,8,-8,-10,-1,-7,-9,-9,7,-7,-3,8,1,-7,1,8,2],[1,-2,3,2,-2,-10,-7,5,2,7,5,-3,-4,6,2,-8,9,8,5,4,8,-6]], dtype = "int8")#candidate|17219|(24, 22)|const|int8
var_17220 = relay.var("var_17220", dtype = "int8", shape = (729,))#candidate|17220|(729,)|var|int8
call_17218 = relay.TupleGetItem(func_5572_call(relay.reshape(const_17219.astype('int8'), [8, 6, 11]), relay.reshape(const_17200.astype('float32'), [792,]), relay.reshape(call_17206.astype('uint16'), [336,]), relay.reshape(var_17220.astype('int8'), [729,]), ), 0)
call_17221 = relay.TupleGetItem(func_5577_call(relay.reshape(const_17219.astype('int8'), [8, 6, 11]), relay.reshape(const_17200.astype('float32'), [792,]), relay.reshape(call_17206.astype('uint16'), [336,]), relay.reshape(var_17220.astype('int8'), [729,]), ), 0)
func_6666_call = mod.get_global_var('func_6666')
func_6668_call = mutated_mod.get_global_var('func_6668')
call_17228 = relay.TupleGetItem(func_6666_call(), 0)
call_17229 = relay.TupleGetItem(func_6668_call(), 0)
uop_17244 = relay.asinh(call_17198.astype('float32')) # shape=(8, 10, 16)
uop_17246 = relay.asinh(call_17201.astype('float32')) # shape=(8, 10, 16)
uop_17247 = relay.log(uop_17244.astype('float64')) # shape=(8, 10, 16)
uop_17249 = relay.log(uop_17246.astype('float64')) # shape=(8, 10, 16)
func_5143_call = mod.get_global_var('func_5143')
func_5144_call = mutated_mod.get_global_var('func_5144')
call_17258 = relay.TupleGetItem(func_5143_call(), 0)
call_17259 = relay.TupleGetItem(func_5144_call(), 0)
output = relay.Tuple([call_17187,const_17199,const_17200,call_17206,call_17218,const_17219,var_17220,call_17228,uop_17247,call_17258,])
output2 = relay.Tuple([call_17188,const_17199,const_17200,call_17207,call_17221,const_17219,var_17220,call_17229,uop_17249,call_17259,])
func_17264 = relay.Function([var_17220,], output)
mod['func_17264'] = func_17264
mod = relay.transform.InferType()(mod)
var_17265 = relay.var("var_17265", dtype = "int8", shape = (729,))#candidate|17265|(729,)|var|int8
output = func_17264(var_17265)
func_17266 = relay.Function([var_17265], output)
mutated_mod['func_17266'] = func_17266
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5200_call = mod.get_global_var('func_5200')
func_5201_call = mutated_mod.get_global_var('func_5201')
call_17288 = relay.TupleGetItem(func_5200_call(), 0)
call_17289 = relay.TupleGetItem(func_5201_call(), 0)
func_11803_call = mod.get_global_var('func_11803')
func_11806_call = mutated_mod.get_global_var('func_11806')
var_17310 = relay.var("var_17310", dtype = "bool", shape = (945,))#candidate|17310|(945,)|var|bool
call_17309 = relay.TupleGetItem(func_11803_call(relay.reshape(var_17310.astype('bool'), [945,])), 3)
call_17311 = relay.TupleGetItem(func_11806_call(relay.reshape(var_17310.astype('bool'), [945,])), 3)
var_17312 = relay.var("var_17312", dtype = "bool", shape = (945,))#candidate|17312|(945,)|var|bool
bop_17313 = relay.minimum(var_17310.astype('uint64'), relay.reshape(var_17312.astype('uint64'), relay.shape_of(var_17310))) # shape=(945,)
output = relay.Tuple([call_17288,call_17309,bop_17313,])
output2 = relay.Tuple([call_17289,call_17311,bop_17313,])
func_17317 = relay.Function([var_17310,var_17312,], output)
mod['func_17317'] = func_17317
mod = relay.transform.InferType()(mod)
var_17318 = relay.var("var_17318", dtype = "bool", shape = (945,))#candidate|17318|(945,)|var|bool
var_17319 = relay.var("var_17319", dtype = "bool", shape = (945,))#candidate|17319|(945,)|var|bool
output = func_17317(var_17318,var_17319,)
func_17320 = relay.Function([var_17318,var_17319,], output)
mutated_mod['func_17320'] = func_17320
mutated_mod = relay.transform.InferType()(mutated_mod)
func_14526_call = mod.get_global_var('func_14526')
func_14528_call = mutated_mod.get_global_var('func_14528')
call_17329 = relay.TupleGetItem(func_14526_call(), 0)
call_17330 = relay.TupleGetItem(func_14528_call(), 0)
func_6968_call = mod.get_global_var('func_6968')
func_6971_call = mutated_mod.get_global_var('func_6971')
var_17359 = relay.var("var_17359", dtype = "float32", shape = (990,))#candidate|17359|(990,)|var|float32
call_17358 = relay.TupleGetItem(func_6968_call(relay.reshape(var_17359.astype('float32'), [6, 15, 11]), relay.reshape(call_17329.astype('float64'), [4, 16]), ), 0)
call_17360 = relay.TupleGetItem(func_6971_call(relay.reshape(var_17359.astype('float32'), [6, 15, 11]), relay.reshape(call_17329.astype('float64'), [4, 16]), ), 0)
func_16619_call = mod.get_global_var('func_16619')
func_16622_call = mutated_mod.get_global_var('func_16622')
const_17363 = relay.const(-6, dtype = "uint8")#candidate|17363|()|const|uint8
call_17362 = relay.TupleGetItem(func_16619_call(relay.reshape(const_17363.astype('uint8'), [])), 0)
call_17364 = relay.TupleGetItem(func_16622_call(relay.reshape(const_17363.astype('uint8'), [])), 0)
output = relay.Tuple([call_17329,call_17358,var_17359,call_17362,const_17363,])
output2 = relay.Tuple([call_17330,call_17360,var_17359,call_17364,const_17363,])
func_17389 = relay.Function([var_17359,], output)
mod['func_17389'] = func_17389
mod = relay.transform.InferType()(mod)
mutated_mod['func_17389'] = func_17389
mutated_mod = relay.transform.InferType()(mutated_mod)
var_17390 = relay.var("var_17390", dtype = "float32", shape = (990,))#candidate|17390|(990,)|var|float32
func_17389_call = mutated_mod.get_global_var('func_17389')
call_17391 = func_17389_call(var_17390)
output = call_17391
func_17392 = relay.Function([var_17390], output)
mutated_mod['func_17392'] = func_17392
mutated_mod = relay.transform.InferType()(mutated_mod)
var_17417 = relay.var("var_17417", dtype = "int64", shape = (9, 7, 14))#candidate|17417|(9, 7, 14)|var|int64
var_17418 = relay.var("var_17418", dtype = "int64", shape = (9, 7, 14))#candidate|17418|(9, 7, 14)|var|int64
bop_17419 = relay.right_shift(var_17417.astype('int64'), relay.reshape(var_17418.astype('int64'), relay.shape_of(var_17417))) # shape=(9, 7, 14)
bop_17430 = relay.subtract(var_17417.astype('int16'), relay.reshape(bop_17419.astype('int16'), relay.shape_of(var_17417))) # shape=(9, 7, 14)
func_16505_call = mod.get_global_var('func_16505')
func_16506_call = mutated_mod.get_global_var('func_16506')
call_17438 = func_16505_call()
call_17439 = func_16505_call()
output = relay.Tuple([bop_17430,call_17438,])
output2 = relay.Tuple([bop_17430,call_17439,])
func_17445 = relay.Function([var_17417,var_17418,], output)
mod['func_17445'] = func_17445
mod = relay.transform.InferType()(mod)
mutated_mod['func_17445'] = func_17445
mutated_mod = relay.transform.InferType()(mutated_mod)
func_17445_call = mutated_mod.get_global_var('func_17445')
var_17447 = relay.var("var_17447", dtype = "int64", shape = (9, 7, 14))#candidate|17447|(9, 7, 14)|var|int64
var_17448 = relay.var("var_17448", dtype = "int64", shape = (9, 7, 14))#candidate|17448|(9, 7, 14)|var|int64
call_17446 = func_17445_call(var_17447,var_17448,)
output = call_17446
func_17449 = relay.Function([var_17447,var_17448,], output)
mutated_mod['func_17449'] = func_17449
mutated_mod = relay.transform.InferType()(mutated_mod)
func_8578_call = mod.get_global_var('func_8578')
func_8579_call = mutated_mod.get_global_var('func_8579')
call_17553 = relay.TupleGetItem(func_8578_call(), 0)
call_17554 = relay.TupleGetItem(func_8579_call(), 0)
output = relay.Tuple([call_17553,])
output2 = relay.Tuple([call_17554,])
func_17585 = relay.Function([], output)
mod['func_17585'] = func_17585
mod = relay.transform.InferType()(mod)
mutated_mod['func_17585'] = func_17585
mutated_mod = relay.transform.InferType()(mutated_mod)
func_17585_call = mutated_mod.get_global_var('func_17585')
call_17586 = func_17585_call()
output = call_17586
func_17587 = relay.Function([], output)
mutated_mod['func_17587'] = func_17587
mutated_mod = relay.transform.InferType()(mutated_mod)
func_12012_call = mod.get_global_var('func_12012')
func_12014_call = mutated_mod.get_global_var('func_12014')
call_17594 = relay.TupleGetItem(func_12012_call(), 1)
call_17595 = relay.TupleGetItem(func_12014_call(), 1)
const_17598 = relay.const([[[-6.344423,-5.182890],[-4.868300,-9.129583],[4.209336,7.921904],[8.781023,8.693396],[-9.552042,-9.087113],[9.521943,1.395431],[-1.377982,-7.503992],[-3.678826,6.779987],[4.018821,5.136859],[-8.500408,-0.143132],[9.489908,6.388121],[5.803505,-1.653748],[8.354332,-3.075848],[-7.945812,6.870269],[9.651218,-9.302957],[2.155460,-5.613783]],[[3.458421,-2.149846],[-8.210441,8.790389],[-6.659740,-0.991476],[7.207337,2.547053],[1.551484,-8.713830],[-7.324825,-8.876054],[3.495700,4.557448],[3.805877,1.023332],[9.506571,-5.368780],[8.006298,7.916539],[-6.601101,0.942307],[-5.077733,-0.952452],[1.417783,4.375450],[-2.696217,6.201244],[-4.561727,-5.101668],[-6.046904,-4.568463]]], dtype = "float64")#candidate|17598|(2, 16, 2)|const|float64
bop_17599 = relay.power(call_17594.astype('float64'), relay.reshape(const_17598.astype('float64'), relay.shape_of(call_17594))) # shape=(2, 16, 2)
bop_17602 = relay.power(call_17595.astype('float64'), relay.reshape(const_17598.astype('float64'), relay.shape_of(call_17595))) # shape=(2, 16, 2)
output = relay.Tuple([bop_17599,])
output2 = relay.Tuple([bop_17602,])
func_17608 = relay.Function([], output)
mod['func_17608'] = func_17608
mod = relay.transform.InferType()(mod)
mutated_mod['func_17608'] = func_17608
mutated_mod = relay.transform.InferType()(mutated_mod)
func_17608_call = mutated_mod.get_global_var('func_17608')
call_17609 = func_17608_call()
output = call_17609
func_17610 = relay.Function([], output)
mutated_mod['func_17610'] = func_17610
mutated_mod = relay.transform.InferType()(mutated_mod)
func_15606_call = mod.get_global_var('func_15606')
func_15608_call = mutated_mod.get_global_var('func_15608')
call_17644 = relay.TupleGetItem(func_15606_call(), 0)
call_17645 = relay.TupleGetItem(func_15608_call(), 0)
output = relay.Tuple([call_17644,])
output2 = relay.Tuple([call_17645,])
func_17649 = relay.Function([], output)
mod['func_17649'] = func_17649
mod = relay.transform.InferType()(mod)
mutated_mod['func_17649'] = func_17649
mutated_mod = relay.transform.InferType()(mutated_mod)
func_17649_call = mutated_mod.get_global_var('func_17649')
call_17650 = func_17649_call()
output = call_17650
func_17651 = relay.Function([], output)
mutated_mod['func_17651'] = func_17651
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10111_call = mod.get_global_var('func_10111')
func_10112_call = mutated_mod.get_global_var('func_10112')
call_17654 = func_10111_call()
call_17655 = func_10111_call()
output = call_17654
output2 = call_17655
func_17694 = relay.Function([], output)
mod['func_17694'] = func_17694
mod = relay.transform.InferType()(mod)
mutated_mod['func_17694'] = func_17694
mutated_mod = relay.transform.InferType()(mutated_mod)
func_17694_call = mutated_mod.get_global_var('func_17694')
call_17695 = func_17694_call()
output = call_17695
func_17696 = relay.Function([], output)
mutated_mod['func_17696'] = func_17696
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13845_call = mod.get_global_var('func_13845')
func_13846_call = mutated_mod.get_global_var('func_13846')
call_17722 = func_13845_call()
call_17723 = func_13845_call()
output = relay.Tuple([call_17722,])
output2 = relay.Tuple([call_17723,])
func_17730 = relay.Function([], output)
mod['func_17730'] = func_17730
mod = relay.transform.InferType()(mod)
output = func_17730()
func_17731 = relay.Function([], output)
mutated_mod['func_17731'] = func_17731
mutated_mod = relay.transform.InferType()(mutated_mod)
func_16505_call = mod.get_global_var('func_16505')
func_16506_call = mutated_mod.get_global_var('func_16506')
call_17732 = func_16505_call()
call_17733 = func_16505_call()
func_9279_call = mod.get_global_var('func_9279')
func_9282_call = mutated_mod.get_global_var('func_9282')
const_17737 = relay.const([-6.515103,-6.721409,7.641035,8.311390,9.906132,0.322095,7.616512,-0.001542,-5.112190,4.928940,2.716126,4.562705,4.190583,-6.132200,5.884774,-5.680092,6.469443,0.622261,-5.559640,6.590272,5.082682,-9.412339,-4.651853,8.228350,-7.933713,5.666203,-5.916434,-2.230202,-9.584314,-4.692595,4.261353,2.377913,-6.370233,5.853075,2.261664,-2.897551,9.879236,0.818016,3.658237,0.353645,4.071249,-1.047745,1.914938,-1.054938,7.436132,3.128359,9.421411,8.404504,9.696667,-0.918707,-9.550734,7.097223,6.307554,7.001568,-2.766957,-7.562195,1.924702,3.900393,-2.646310,-1.877834,5.980487,-1.624811,-3.705844,-9.842643,-4.607596,2.412448,7.559219,6.250903,2.031211,-3.638546,4.655532,-7.992621,0.039517,8.620710,0.310612,1.229499,6.387290,7.031678,3.181821,-4.107898,-2.443876,-7.500965,-1.027054,-2.167310,-6.346130,6.956421,-3.962284,1.986309,-7.611591,1.870844,-9.955617,-8.172926,-8.594907,-2.682698,6.580728,6.135777,-1.151393,4.185880,8.381115,-7.071856,-1.964957,9.131053,9.176864,-2.800095,8.447879,8.835724,-2.306658,-5.842491,1.942944,9.190434,4.743118,-8.898751,-5.535849,-1.200647,-1.353759,-1.677650,9.236968,8.084579,-6.091418,-3.556752,5.374834,6.557189,8.484230,-1.679468,-1.721811,0.287702,-6.114925,4.351777,9.384708,2.585533,6.901400,8.571812,8.302023,7.601828,-1.502767,-9.205839,-8.571250,-8.410691,6.165206,-9.386887,-2.767759,2.704424,-0.173276,6.332997,-3.906875,5.263532,-6.894948,-2.225923,1.173361,7.614366,8.696883,1.481744,-7.107556,9.905150,-6.150769,3.677692,3.382733,1.389821,-7.509656,-0.799289,5.162624,-4.247286,-0.297434,-7.057851,-9.051709,7.134938,2.354903,3.953072,-8.085917,9.809248,8.163012,6.343381,4.541729,-1.821708,-5.589842,-0.500836,-4.160826,7.850697,-7.637461,-3.267921,-0.117341,-7.512622,-0.538803,0.511535,4.574274,8.823019,8.465767,1.885480,9.941432,-4.859659,-9.279611,-3.637304,0.844640,7.239968,4.225816,2.202606,-5.226319,0.342037,7.173803,9.706573,7.068617,1.227427,-3.344229,3.182012,-0.718572,2.710428,-8.053448,5.546437,1.248815,7.096689,-2.180474,-7.353805,4.999572,6.577545,-3.387881,3.711688,8.153827,8.313577,6.569022,-4.614848,-9.162144,0.993304,1.895571,6.409037,6.188355,7.589594,5.438261,-8.054509,0.457346,6.738439,4.578628,2.261403,-8.683281,-4.026800,7.755051,1.150362,-1.764176,-1.976966,-8.030704,-9.979942,3.029992,6.287667,-7.023551,9.490037,-4.609542,9.884411,2.575415,-0.379358,2.845414,0.755910,4.220068,2.194983,1.818845,6.664126,5.201642,-4.442530,3.076181,9.333390,3.636843,8.611358,0.105501,4.717710,8.974365,-0.024487,-0.392609,3.662574,6.777042,6.751036,3.843670,-6.060741,8.032827,1.754610,-3.793533,-6.088731,7.710188,4.230843,-9.964128,5.306079,7.435955,6.611299,5.682258,7.869511,8.473456,-7.942455,-2.854969,1.802261,4.189321,-1.554142,1.957253,5.599827,0.052992,4.460979,-2.118585,8.646264,8.881137,6.340451,-0.387891,3.355421,-2.357310,-8.502828,-5.127245,-1.472410,-2.614730,4.407734,6.479998,-8.049442,-3.603117,-3.890222,-2.243387,-0.143482,-1.174848,-4.537432,-8.744249,-9.717449,5.138801,7.663884,-4.490190,3.964511,6.986168,-8.750652,-1.474469,6.730825,6.992737,4.504724,3.303824,4.196633,0.883382,-4.574810,4.625402,-3.808582,9.949009,-1.197377,9.296144,-2.213555,-1.399903,6.492962,9.646061,4.259338,1.963252,-6.936972,-0.786604,-2.904210,-6.074684,-6.186979,-1.118870,-2.079058,-7.318111,5.754584,0.871721,-8.155428,5.683914,-7.988908,-8.847026,5.272166,-3.166398,9.208107,4.186988,-7.221601,-2.596331,6.229472,-0.991075,5.601600,-6.293062,-6.598395,-5.971963,0.682084,9.855153,3.113805,-4.319820,0.496493,-2.500990,8.421848,-1.442230,2.067540,-9.305617,8.397276,3.285394,-6.140800,-1.980710,0.427237,2.560690,-5.996379,-2.119594,3.733499,-9.523476,0.113242,0.799132,-3.400847,-8.589963,-2.358858,-2.684191,4.082202,-5.527279,2.570663,4.220204,-3.700529,2.266865,8.036556,-4.007106,-1.311735,4.959252,5.618887,8.703230,-1.208686,-9.665712,-0.405916,2.446318,-5.961391,9.625207,-5.551118,-2.819868,-1.234884,6.109016,9.790595,-1.392471,-1.556788,2.765955,-9.652892,4.071956,-8.529401,2.668106,-0.230228,1.615252,2.170728,-1.993382,5.474607,-0.355052,-9.800918,2.753472,7.364841,-1.780256,8.284349], dtype = "float32")#candidate|17737|(432,)|const|float32
call_17736 = relay.TupleGetItem(func_9279_call(relay.reshape(const_17737.astype('float32'), [12, 6, 6])), 0)
call_17738 = relay.TupleGetItem(func_9282_call(relay.reshape(const_17737.astype('float32'), [12, 6, 6])), 0)
output = relay.Tuple([call_17732,call_17736,const_17737,])
output2 = relay.Tuple([call_17733,call_17738,const_17737,])
func_17739 = relay.Function([], output)
mod['func_17739'] = func_17739
mod = relay.transform.InferType()(mod)
output = func_17739()
func_17740 = relay.Function([], output)
mutated_mod['func_17740'] = func_17740
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11879_call = mod.get_global_var('func_11879')
func_11880_call = mutated_mod.get_global_var('func_11880')
call_17764 = func_11879_call()
call_17765 = func_11879_call()
func_8684_call = mod.get_global_var('func_8684')
func_8687_call = mutated_mod.get_global_var('func_8687')
const_17768 = relay.const([True,False,True,False,True,False,True,True,True,False,False,False,False,True,True,True,True,True,True,True,False,False,True,False,True,True,True,False,False,True,True,True,True,False,False,False,False,True,True,True,True,True,True,False,True,True,True,False,False,True,True,False,True,True,False,False,False,False,True,False,True,False,True,True,False,True,True,False,False,True,True,False,False,True,False,True,False,True,False,False,False,True,False,False,False,False,False,False,False,True,False,False,True,False,True,True,True,False,True,False,True,True,False,False,False,False,True,False,True,True,False,False,True,False,False,True,False,False,False,True,False,False,True,True,False,False,True,False,False,False,False,False,True,False,False,True,False,True,True,True,True,True,True,False,True,True,True,False,False,True,True,False,False,False,True,False,False,False,False,False,True,False,False,False,False,False,False,False,False,True,False,True,True,True,True,False,True,False,False,True,False,True,True,False,True,False,False,True,True,True,False,False,True,False,False,False,True,False,False,True,False,False,False,True,False,True,True,True,True,False,False,True,True,True,True,True,True,False,False,False,True,False,False,False,True,True,False,False,True,True,True,True,True,True,False,True,False,False,False,True,False,False,True,True,True,False,False,True,False,True,True,True,True,False,False,False,True,False,False,False,True,True,False,True,False,False,False,False,False,True,False,False,True,True,False,True,True,False,False,False,True,True,True,True,True,True,True,False,False,False,True,True,True,True,False,False,True,True,False,False,False,False,False,True,True,False,False,True,True,False,True,False,False,False,True,True,False,True,False,False,True,True,True,False,False,False,False,True,True,True,True,True,True,False,False,True,False,True,False,True,True,True,False,True,True,True,True,True,False,True,True,True,False,True,False,False,True,False,True,False,True,True,True,True,True,True,False,True,False,False,True,False,True,True,True,False,False,False,False,True,True,True,False,True,False,False,True,True,True,True,True,False,False,False,True,True,False,True,False,False,True,True,False,False,False,False,True,True,False,True,False,False,False,False,True,True,True,True,False,False,False,False,False,False,False,False,True,False,True,True,True,True,True,True,True,True,True,False,False,False,True,False,False,False,True,True,True,False,False,True,False,False,True,False,False,False,False,True,False,False,True,True,True,False,True,False,True,False,True,False,False,False,False,False,False,True,False,True,True,False,True,True,True,False,False,True,False,False,True,True,False,False,False,True,False,True,False,True,False,True,True,False,True,False,True,False,True,True,True,True,True,False,False,False,True,False,False,True,True,True,False,True,True,True,True,True,False,True,False,False,False,False,True,True,False,False,True,True,True,True,False,False,True,True,True,False,True,True,False,False,False,False,False,True,True,True,False,False,False,False,False,False,False,True,True,False,True,True,False,False,False,False,True,True,True,False,False,True,True,False,True,True,False,True,False,False,False,False,False,False,False,False,False,True,False,False,True,True,True,True,False,True,False,True,False,True,True,False,False,False,False,False,True,True,True,True,True,False,True,True,False,True,True,True,False,True,True,False,True,False,True,True,True,True,False,True,True,True,False,True,True,False,True,False,True,True,True,False,True,False,True,True,True,True,False,False,False,True,False,True,True,True,False,False,True,False,False,False,True,False,True,False,False,True,False,False,False,True,False,True,True,False,False,True,True,True,True,True,False,True,True,False,False,True,False,True,False,False,True,True,False,True,False,True,False,False,True,False,False,True,True,True,True,True,False,False,False,True,False,True], dtype = "bool")#candidate|17768|(720,)|const|bool
const_17769 = relay.const([[-9.837213,0.844834,-7.825213,0.427589,4.821268,7.477785,6.361382,7.892409,4.705105,-5.146589,7.792843,-9.171772,-7.730492,8.186087,9.273451,-8.605592,4.031712,7.439625,5.727716,4.608259,-4.423330,-9.976051,9.056156,6.727571,4.071023,8.009447,8.240705,-6.559780,2.719900,-7.318406,6.188832,-9.132931,-2.298451,-9.000443,8.249967,4.638401,-6.820712,9.132831,8.012864,4.308106,-2.493715,0.154969,-9.444621,5.861855,2.819868,6.776340,-3.998784,-8.651843,5.171639,-3.684265,-5.468399,-4.752535,8.798961,-4.028506,-4.422608,2.841819,-5.908512,-5.243838,-9.404696,2.596105,4.247848,-6.283492,-1.719930,6.971374,-1.107242,-8.350614,3.542935,8.989302,-1.551484,8.895114,0.775520,0.739758,0.556705,2.815632,0.415382,2.466306,8.671882,-2.467892,-7.963071,1.923967,2.034390,2.496533,-9.650585,-1.034120,4.701245,-2.133451,-8.818900,-9.616951,2.648721,-9.887041,-1.292680,-0.792498,8.236318,-9.588661,-6.016733,-4.510566,3.047302,-3.403043,5.529388,4.236551,-0.542550,2.152973,0.630964,-3.231665,8.593467,-5.722748,-4.504832,5.059059,3.582079,-2.033450,3.879950,-5.222030,-2.022455,3.354577,9.015668,-6.234746,-6.977665,-5.768634,-4.197754,-0.368892,6.993502,6.168275,9.442249,-2.105809,-9.372589,-8.597105,0.422703,3.180240,8.945205,0.917762,9.646105,-7.140207,-6.782231,0.035749,0.319255,-2.593242,-1.538854,4.232144,2.511204,9.788927,-3.134939,0.060015,-4.878072,3.276654,3.839179,-8.957250,3.676940,-9.504608,8.807451,-7.003693,-4.717655,-9.162693,-9.137459,-6.828147,7.040393,-4.743331,-1.970572,6.965625,-1.272921,-9.655883],[-1.791824,-1.874159,-6.746570,8.195692,-7.680591,-1.068513,-6.237406,-3.745933,-1.948598,-3.040527,-5.000154,5.349741,4.014151,8.806593,-1.600913,8.031337,-8.717260,-6.008497,5.373540,-6.757249,6.629557,-0.431653,1.700810,-6.954420,2.001622,8.030185,9.355381,-7.768152,-4.154633,-3.025709,-2.721240,6.644770,5.994536,5.913226,3.759038,5.134686,-6.172000,-6.372016,-7.001039,-0.429331,-0.940555,3.150801,-3.822879,-3.733615,6.141886,4.226406,-6.705297,-4.667232,-7.772992,-9.575057,6.277132,-4.228787,1.317471,-3.797756,-1.347190,-7.467045,-6.434788,-6.758382,2.648209,0.470109,-6.634717,-4.870029,6.189282,-8.198309,-7.112099,3.137601,-6.183548,-6.349535,3.474560,5.432719,8.579470,-8.640913,8.106878,-6.624101,5.355448,7.838046,0.307965,-2.531225,-2.605311,-4.599208,-9.281909,1.281840,-5.900433,-3.326911,-0.059727,9.725229,1.633931,4.661984,-2.651094,-7.459026,-4.284266,5.854746,2.351140,-7.451482,6.731543,5.906424,8.389347,-3.942419,-4.837592,2.020951,1.606222,-7.149350,-3.014766,7.082278,5.406226,-0.815892,-4.558575,4.347298,4.395750,2.232501,9.591827,-8.238302,0.073893,-7.806516,-6.961628,-2.026253,-8.727761,-7.250982,9.684608,6.806752,9.722971,-9.415755,-7.810001,-9.497140,4.427098,-9.690303,-2.263716,2.785264,2.677631,1.092843,4.662532,4.375061,-6.735607,-9.889586,-2.376487,-0.965284,7.365721,2.498457,7.450719,-5.819886,2.091249,1.841781,5.750198,-8.660479,-5.804234,-5.655094,6.520691,3.450734,-2.699407,2.139672,6.485495,-6.189183,-2.246782,-1.935761,8.098687,9.408519,3.555663,-5.979242,9.773603,-6.243195],[-7.076871,-8.230189,-4.985051,2.623040,8.331957,4.430386,4.753761,-1.098506,1.356789,2.592563,5.512441,-6.429181,6.513615,3.825108,1.098391,-4.213512,4.305309,3.664913,-7.043371,-9.363132,-8.760109,-9.782400,0.351111,3.815686,-1.390700,-4.248134,6.461975,-6.349874,2.228787,-2.267897,0.493440,0.424246,6.749202,3.711974,8.433605,4.872061,8.319349,5.409882,-3.724075,-8.829694,4.086468,-7.172224,8.339603,7.308763,1.598550,2.749978,-4.773786,6.466714,3.685267,1.451481,-9.525722,-7.462337,3.226004,-9.206403,6.366584,-5.616628,3.078687,2.876960,1.912354,-4.038005,-3.926519,-6.820930,-5.878306,-6.455855,-2.880294,-1.505135,8.295247,-7.718396,1.288704,-3.890009,4.267949,7.062062,-9.209894,3.109819,-5.013465,-9.079891,9.273852,5.434859,-3.986788,-9.761959,-5.020394,4.479031,3.791925,-3.401957,-8.809736,3.230705,7.245030,4.860193,-3.092704,9.609119,-2.559741,2.829400,-4.678595,4.572591,4.669412,9.232402,-4.311175,9.925145,1.073584,-9.044043,-4.523309,2.223139,8.220762,8.138411,-0.748637,-8.129365,-1.768695,-2.687214,-0.809276,9.311782,7.794873,1.657079,9.459161,-2.651266,-2.934899,1.565410,-0.490072,-1.766768,3.803838,6.070814,2.538761,-2.556135,-2.690741,-9.315351,-3.236032,-5.459350,5.083594,1.676228,-8.359619,0.540715,8.384345,1.535703,6.944745,4.437115,-2.484393,-5.100726,-3.340458,6.106558,-1.067555,4.729696,-9.804770,-7.137967,-1.913064,-4.128698,4.902963,2.647408,-4.809233,5.832016,8.469471,-9.053305,9.940584,1.639232,9.076355,-2.246211,-0.857442,-0.507987,0.855884,-2.954989,8.838399,-0.636786],[-8.423949,3.222116,5.645893,9.251623,9.415572,-6.248109,-9.273179,-7.956666,-2.670207,8.121183,-9.286640,6.359119,2.349815,9.114493,4.817076,1.662718,-8.152880,-2.163167,5.241171,-7.125808,-0.081632,-6.634339,-4.502187,-5.752628,8.973971,0.482336,-8.413522,-1.428195,6.427079,-6.609786,7.115144,-1.494019,-3.906544,6.573550,-0.559963,-2.925068,-6.993281,-9.944828,6.415839,-2.803778,-3.595140,-5.847602,-4.868796,-6.482441,1.586199,-2.404186,9.521767,-4.153789,-6.839130,8.092904,6.687283,2.176053,-1.131801,-6.350318,-1.078310,8.072086,-9.412741,1.559983,-2.108727,-1.998148,-0.357970,-7.864743,-8.384372,6.559933,6.530286,4.076687,-0.878943,-7.723399,-6.351362,-4.985951,6.048664,-0.633835,-8.215102,0.800636,6.665061,-3.382076,-4.322149,7.392095,5.076252,3.163514,-5.286253,8.319639,0.259932,4.413892,-8.882741,7.536770,2.963586,0.216506,2.354358,-1.635307,5.262397,3.692715,-3.294925,-2.627057,0.433508,-3.040442,-5.304110,1.621862,2.243099,-0.630667,-5.468559,-7.036241,2.320282,7.855726,-7.125705,9.454065,-1.883838,8.943888,-2.446764,-9.562492,-5.088406,2.779211,-3.384488,-5.797743,0.021020,6.277145,-2.949207,-0.005682,-7.319326,0.728323,1.806416,-9.965223,7.615961,-6.437644,4.932749,9.685320,9.132982,-1.410018,8.074160,-3.786191,9.470026,4.481638,8.767452,4.336726,0.981510,-0.835288,-1.240728,1.893163,-9.320413,7.580625,-7.740648,3.841289,6.944717,-1.903616,7.593491,3.930918,-7.821277,-1.327972,-3.441252,2.731866,-9.656551,2.177657,-1.268037,-6.933203,-4.133541,-3.670941,2.188694,8.129784,-2.185172,1.215965],[-2.593896,-1.339566,1.496012,-3.531289,1.056477,-6.909154,0.467869,-9.212232,-8.972613,-9.811047,-2.215198,3.901178,0.275517,-5.008053,-4.858101,6.788046,-8.168124,-0.190542,-2.597049,-2.532964,-3.759413,-6.478774,-6.159481,-0.858332,9.716203,9.769436,9.749740,4.833842,7.892644,-2.671669,-8.752260,-8.440608,-7.338694,-6.817213,1.918724,-7.110388,2.616163,2.898343,8.328556,-4.233140,2.077042,-6.717863,1.956123,7.368754,0.705790,-5.966803,4.720936,-2.267754,1.732929,-1.444748,7.885404,7.763860,4.011145,-1.676237,-7.469067,-9.437111,7.121932,-7.672652,-5.718430,5.963711,-0.075096,6.013420,8.218988,0.463772,9.384011,8.166246,-7.894972,7.385306,-2.079914,-4.092724,7.347343,-5.071084,1.471577,9.848928,-1.580283,7.271014,-9.848107,-9.615537,-4.005326,6.662382,0.594719,4.174946,4.699951,-2.527714,-7.587875,-5.848575,4.630349,-9.757145,9.511330,-4.895959,1.927364,-1.458865,8.538473,4.162915,-8.559412,7.516262,6.278939,8.916883,1.769291,-5.879757,-3.040494,5.724783,4.052644,-4.612376,7.397059,-8.923776,-7.336803,4.300535,4.865144,-0.624567,6.001399,-6.358073,-0.817914,8.132669,8.893059,0.521439,-4.207752,4.830318,8.380133,-9.765581,-6.381618,3.243170,-2.172283,-6.607593,2.306219,3.789595,-2.396111,3.270438,-9.557724,0.315164,2.089707,9.312614,1.429049,5.039067,6.717480,-2.666496,6.637847,-9.150408,-5.162423,8.690773,7.296743,-6.264325,-0.694751,-4.442391,-7.186969,-3.610708,3.223366,7.950956,-5.855657,0.696524,-8.147577,-4.575020,4.278216,3.070429,3.353011,6.256553,-8.157361,9.825979,8.766367,5.678954],[6.773863,7.432601,1.051439,2.946760,-8.084398,1.807995,-8.473744,-9.708559,5.542895,5.493092,-3.352163,9.355190,-5.789931,-9.101715,0.482859,9.359235,0.969103,0.558743,2.043923,3.398375,7.077932,4.440645,-1.089801,9.289517,2.037226,-8.860566,4.289504,-7.957548,-4.124384,0.231726,-0.735412,-8.218327,-1.889672,5.003968,0.024191,1.165141,0.583392,-1.634353,-5.676554,1.463797,0.789740,5.612210,-8.862914,-0.434812,-1.023804,-3.330176,4.591376,6.092222,-6.261471,0.270667,7.433042,-2.056515,3.070054,-7.536650,-1.587538,-8.021896,8.848569,-3.005372,6.003495,4.858064,-0.958202,3.273436,-4.855397,4.079698,2.530918,-3.044570,5.063763,7.549835,-5.130197,-6.750934,-6.254855,-5.567240,1.463024,4.364849,0.844740,-0.447821,-1.623620,7.754202,-5.295305,9.765502,-0.597017,-8.855918,-6.111331,-4.231343,5.810512,6.450142,5.390379,-7.104942,-8.202031,-0.061689,-9.502597,9.302070,-5.439428,2.086031,-5.204526,-8.426851,3.135229,-1.286821,-9.375321,-1.207480,-2.029795,6.379773,4.782508,-1.727936,0.752243,8.770346,9.712263,-9.642780,-5.575842,7.276313,-5.451853,3.471158,-7.615734,7.140744,-8.775723,0.065454,8.133302,2.968501,-8.954433,-0.539772,-8.151852,1.315162,4.278913,5.908320,4.103427,-8.944120,-6.575155,0.612649,1.411845,6.576527,-8.847092,-3.876286,-6.205642,-7.407596,2.231563,-8.504570,4.767520,-6.165236,-6.506163,-8.444004,-2.906875,2.662092,-8.478309,1.598789,-6.217697,8.957558,0.976972,-3.653257,5.602951,-1.645733,-9.972191,-3.903237,-5.572581,-6.857926,-8.621076,1.234496,2.199581,6.639854,6.952452,6.778584],[-2.425471,7.103047,7.064678,6.835285,-0.492850,6.916823,-0.404843,-7.193732,-0.369471,5.392225,1.397638,4.139783,-7.812261,-9.041812,-6.169783,5.856220,2.782302,-5.849647,-0.164088,6.947896,-0.455477,2.231320,0.570500,4.152596,-1.896543,-4.306088,6.115249,-1.834296,1.962558,-5.795891,9.294705,-3.840743,-0.299772,0.757345,-9.230860,4.226241,2.832215,9.768119,1.790303,7.456126,-2.284102,5.863499,3.484413,2.156096,9.998542,-1.828995,8.848249,9.534412,-4.319614,-3.901104,7.242820,9.221879,8.788276,3.082350,8.082352,-5.532310,2.530946,-8.785387,-4.030098,5.546479,9.771347,-4.651310,-7.214425,-1.433451,-0.566153,4.076659,-3.420203,-4.839409,7.723663,-2.187592,-6.294646,-1.564163,7.803346,3.173239,-6.992802,3.815059,-6.485082,-6.409221,2.894763,9.043289,-6.394081,3.658219,0.409996,-5.610628,-8.006549,-3.055925,1.736990,-5.076026,1.653165,-0.925722,-1.692279,3.873048,3.857591,2.758640,-7.017694,-8.578299,5.032184,2.475091,-5.597974,2.364206,8.082719,6.371136,3.891720,3.755937,9.696200,-4.606925,2.627955,7.534031,-9.403980,4.083060,8.386668,0.004996,8.898854,-3.663622,-3.750848,-2.831121,5.258579,-6.110800,2.131242,1.013274,-2.857758,-8.594561,3.865565,-2.924302,-6.919815,-6.200942,-7.845552,-6.450244,-2.207666,9.020507,7.582298,-8.905935,3.028379,9.208131,-3.289081,-4.170098,-1.230519,-5.252175,-5.422386,9.141139,-1.117766,-4.130331,0.479515,0.973485,3.158274,-6.065510,-8.707949,8.332057,-1.348910,3.202044,4.028706,1.747178,-3.438764,5.402097,-2.528353,-4.911690,-0.673091,3.908861,9.282008,-0.890930],[1.509343,4.929452,8.246953,-4.849263,3.229428,4.657186,8.409747,4.549067,-3.277212,-4.343054,9.992563,-2.251645,0.119622,-9.975964,-2.775717,-5.511172,4.730231,4.204792,-3.372358,-3.316162,7.457398,2.552816,-2.519093,4.778034,6.859578,-6.948491,-0.383777,5.994785,-1.539278,-4.393668,-8.657072,9.317309,9.981006,-3.442397,-1.128363,5.897019,-1.089913,-5.708064,3.115792,-6.358193,1.330216,-0.266932,6.669963,3.230841,7.438901,-9.218133,-0.671617,9.791099,-4.541389,-8.119081,6.333620,5.274425,-4.767153,-1.872087,1.092577,-6.593798,4.237551,-8.690242,-6.627161,4.750699,0.040033,6.959460,3.169149,9.994728,1.866387,-0.787774,-3.706706,-6.291223,0.847943,-5.081468,6.125252,3.584696,6.788069,2.331334,0.277341,7.998810,-1.443406,5.203100,-3.720464,-8.692558,-7.623752,-2.510479,-3.685769,4.116961,8.462439,-4.607098,-9.017435,-6.000619,1.741423,-0.329862,-2.123644,6.466438,-5.915461,-1.510335,-6.881004,-5.964683,3.466466,6.371875,-4.559747,1.193545,-5.247896,-7.877448,2.157713,8.164464,4.512227,2.890098,2.519909,-4.247727,9.252051,-8.447569,8.430214,-9.973498,0.068246,-1.968020,-6.200648,-7.998432,3.974611,-2.503487,7.616540,2.640296,9.673170,-7.750744,-6.707951,-6.715751,1.362453,0.342897,-3.152193,2.415422,9.270723,3.288410,2.938161,-1.079589,-3.242378,3.996235,-7.844404,-3.757081,9.946508,1.411802,-5.011134,-0.510106,-1.510028,-3.520393,8.393130,6.414093,-0.537085,3.438083,-7.559863,-0.085785,-5.212611,-2.522029,2.545016,-6.351178,3.073033,-2.777597,8.394346,7.447362,-4.043518,-1.194105,4.187621,-8.301119]], dtype = "float64")#candidate|17769|(8, 160)|const|float64
call_17767 = relay.TupleGetItem(func_8684_call(relay.reshape(const_17768.astype('bool'), [720,]), relay.reshape(const_17769.astype('float64'), [1280,]), ), 0)
call_17770 = relay.TupleGetItem(func_8687_call(relay.reshape(const_17768.astype('bool'), [720,]), relay.reshape(const_17769.astype('float64'), [1280,]), ), 0)
output = relay.Tuple([call_17764,call_17767,const_17768,const_17769,])
output2 = relay.Tuple([call_17765,call_17770,const_17768,const_17769,])
func_17774 = relay.Function([], output)
mod['func_17774'] = func_17774
mod = relay.transform.InferType()(mod)
output = func_17774()
func_17775 = relay.Function([], output)
mutated_mod['func_17775'] = func_17775
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10383_call = mod.get_global_var('func_10383')
func_10384_call = mutated_mod.get_global_var('func_10384')
call_17820 = relay.TupleGetItem(func_10383_call(), 0)
call_17821 = relay.TupleGetItem(func_10384_call(), 0)
func_12507_call = mod.get_global_var('func_12507')
func_12508_call = mutated_mod.get_global_var('func_12508')
call_17822 = func_12507_call()
call_17823 = func_12507_call()
output = relay.Tuple([call_17820,call_17822,])
output2 = relay.Tuple([call_17821,call_17823,])
func_17824 = relay.Function([], output)
mod['func_17824'] = func_17824
mod = relay.transform.InferType()(mod)
mutated_mod['func_17824'] = func_17824
mutated_mod = relay.transform.InferType()(mutated_mod)
func_17824_call = mutated_mod.get_global_var('func_17824')
call_17825 = func_17824_call()
output = call_17825
func_17826 = relay.Function([], output)
mutated_mod['func_17826'] = func_17826
mutated_mod = relay.transform.InferType()(mutated_mod)
func_11958_call = mod.get_global_var('func_11958')
func_11960_call = mutated_mod.get_global_var('func_11960')
call_17839 = relay.TupleGetItem(func_11958_call(), 0)
call_17840 = relay.TupleGetItem(func_11960_call(), 0)
func_5513_call = mod.get_global_var('func_5513')
func_5516_call = mutated_mod.get_global_var('func_5516')
const_17848 = relay.const([[-0.436892,-1.020938,9.563916,-4.254684,4.030269,3.585473,-7.318525,5.699491,3.204402,-8.732124,-1.022625,-5.162348,3.038627,4.560221,-4.612160,3.149315,-8.581425,9.160095,-8.784753,7.016663,-2.324749,8.221934,9.418011,2.995653,-4.527167,7.602247,9.445540,-9.287202,-9.759179,-7.357908,-6.564572,1.414809,-2.047160,-1.015825,-1.166786,-0.330962,0.968696,3.506474,-3.106629,8.090525,-5.509512,3.445290,6.226472,-3.384573,-8.743889,-6.405220,5.796561,1.561806,-5.955776,-0.088516,8.582835,-6.107916,-8.436457,7.570955,-0.042316,6.099500,-4.053841,-1.812877,4.724111,3.471071,-6.708846,-9.677152,1.583182,0.421374,-9.389022,-2.909681,-4.040738,-9.239782,-4.996822,-0.559224,-3.910006,5.504510,-7.434997,-2.368092,-7.745103,4.158513,-1.426192,-7.907518,7.033154,8.656471,5.312407,8.288795,-6.400662,1.416572,-2.270809,3.199408,7.578189,5.283188,2.315184,4.537490,4.179525,-6.061701,7.798877,3.547432,-7.145106,-3.023806,-9.577044,1.899991,2.841104,3.263508,-6.382669,-4.651117,0.600835,-6.017381,-2.750176,-7.963524,-9.419987,0.366875,-6.427474,8.656754,3.710388,-0.756217,2.279960,5.500943,-3.733267,-0.774074,5.734061,5.687223,-6.449955,8.399477,9.116159,-1.989176,-0.206785,5.776150,-7.697166,1.625142,-9.221601,5.856510,-9.315220,-2.608382,0.032172,2.651017,-5.651084,3.919081,-9.455317,8.708131,0.020801,-6.386367,8.810942,-0.940839,-1.831883,7.536712,7.240654,-7.641803,1.484227,2.208498,-1.194179,-0.495879,4.269493,9.222021,-7.696291,1.420496,9.185877,-2.860539,-3.659971,-7.295865,-7.642026,3.758214,9.607556,4.252508,6.225094,9.433606,-6.568283,-9.079213,-2.933991,9.686830,-0.480246,-8.320016,-4.323163,-8.387376,0.898581,3.353559,-1.509965,0.430897,0.467553,2.873105,7.705039,-8.338935,-6.620741,2.272875,-5.540019,4.677297,-8.236830,-8.713986,-0.887275,3.433081,-8.835331,-3.796743,4.716570,-5.084709,-1.094559,5.810723,-5.555197,8.391828,-1.526537,4.670987,-8.244569,8.110909,-7.510170,1.675405,-2.656138,-8.991717,-8.869085,-8.706432,3.541331,-5.934451,-2.529944,9.262148,8.120222,5.434178,-6.310071,-8.634611,8.698090,8.056522,-6.770002,2.338259,4.869664,3.285968,-1.717097,4.324879,-5.744007,0.436593,-5.552932,3.143182,-0.091636,9.122185,-6.270982,7.988588,-6.537952,6.474106,8.360538,3.078663,4.460809,-5.736296,-1.226906,9.033892,2.583928,7.522778,4.301802,4.810648,6.484359,-3.290261,0.466349,-1.768265,-3.272505,-0.217943,-3.521351,-1.283372,-0.452392,3.199791,4.799916,0.643146,7.014434,-1.690178,0.334648,-5.691167,-2.444816,-8.483028,-4.494755,9.955269,1.039441,7.486693,-3.159929,8.044833,-5.309309,9.370048,7.642699,7.190087,-2.781203,5.222095,2.130627,-0.851642,7.018430,6.933455,9.468666,-3.751704,-9.250937,-6.710519,-8.745906,0.683576,-7.028060,-2.655754,6.183593,2.954378,3.039651,-1.012455,-9.251944,-8.399986,0.543996,-0.912940,2.615142,4.058267,-2.359593,-5.358735,3.533837,7.362852,-7.264560,7.192853,7.029168,4.393856,-3.567383,-2.437030,1.478547,-9.176057,-4.507786,8.956939,0.331771,-2.009493,9.329907,-8.303666,-3.729815,-5.736212,1.084083,6.646947,-8.155333,3.976264,-8.763608,-6.830727,0.140102,-7.194067,0.804413,6.265296,6.452306,-0.893357,-3.321311,6.071624,-9.125497,5.492050,-8.051668,3.746722,-1.176823,7.860306,-1.625093,-4.072144,-7.037703,0.924245,-5.147632,-1.499293,-5.169547,-4.327371,-0.256872,-2.627175,-7.640618,-6.361748,-9.336651,1.139785,2.811339,1.611213,-4.539217,1.431294,-1.699421,-6.765335,8.398300,-4.608302,2.497286,-4.575354,5.849092,5.276911,-1.437981,-0.703041,4.832190,3.616023,-7.300581,-7.090394,-2.921562,0.103287,-4.466483,-2.794427,-9.677708,-6.525694,6.435039,2.131648,5.904069,-2.352047,0.995679,-8.892190,7.302691,6.951943,-0.947868,-9.830465,6.807430,6.686145,-4.051768,0.885426,9.271484,-4.536251,-0.902500,6.072721,-0.673839,3.197940,-9.374139,-5.232554,-3.922578,2.458393,9.675717,5.990887,6.878368,5.018450,-3.200169,-6.702884,-3.048785,5.864988,9.456776,-4.702301,-4.551896,2.675353,4.699121,-5.612547,3.434576,-1.210307,-2.350665,1.481112,-4.393755,8.073808,8.395991,-5.992384,-6.021177,-0.895354,-3.394185,-8.046190,-8.023152,2.352453,5.372892,4.891748,2.881879,-1.399074,-3.796241,3.629851,-4.081932,7.340651,0.101346,2.521202,6.663120,1.307358,-5.853854,2.131650,7.754818,-7.510125,-3.119759,3.247924,0.864127,4.212364,6.105118,-5.507399,0.817931,4.287272,-9.880108,3.908398,-6.764801,9.912175,-8.461749,0.881252,-1.668270,-4.794366,-3.625307,-9.757441,0.563858,4.176839,9.177246,-0.972276,1.995046,8.887758,-5.404503,-7.736229,7.833390,5.180782,-9.592841,7.314680,-4.072514,3.399936,1.338635,-2.325983,-3.689414,8.962681,-5.517912,-6.414160,0.168336,9.919341,-2.207251,-2.587961,6.322668,0.842259,-3.291261,-0.412783,9.360510,6.180357,-8.431574,0.678332,4.768994,-3.323926,4.328119,-3.785139,-5.176891,-7.549493,2.199394,-9.857333,3.830800,-4.052454,-2.488142,-7.348972,-8.336409,-3.295296,-7.194949,5.633767,-1.280977,9.464463,0.072996,-1.333423,0.610421,-5.852628,0.349246,1.364412,7.147621,-9.543249,5.458835,8.388163,9.917831,-7.246343,5.217660,6.955088,2.394677,4.318253,-9.371261,-6.690569,8.836595,-0.391199,0.362696,7.384277,5.536064,-2.560811,-0.808960,2.280368,4.462549,6.349219,8.660450,-6.346043,6.363218,8.281410,5.151323,7.690452,7.094724,2.914752,-9.211296,3.067438,-1.921500,-8.413535,2.344161,-9.861022,8.633436,2.061218,1.303714,3.531372,-2.858854,-8.835612,-1.994764,-8.360679,-7.557655,-8.956701,-2.496859,-2.067729,7.510112,2.302487,-0.461985,7.449819,-1.538314,3.927346,-0.943804,-2.652949,6.678157,-4.655292,6.692720,7.368709,-7.826301,-9.838801,6.116262,6.800231,0.370546,2.281887,4.476368,5.784001,-4.038377,8.681664,6.400032,6.467216,-1.837724,-7.684443,-9.807173,7.319542,6.598378,1.570173,5.077904,6.205202,0.390076,-9.506212,1.955280,-3.330245,-5.285397,-8.682691,-2.330106,5.136923,2.020054,-0.631127,8.298303,-5.280632,5.027429,-8.506341,-0.705240,3.463217,-0.434236,-2.624662,1.868813,0.105457,-1.176221,3.662730,1.607109,-5.790656,-3.335527,-7.448168,-0.548122,-0.680614,-0.853443,-5.828787,-4.130216,8.860527,-1.522816,-4.357621,-9.385750,7.062416,2.199232,1.477271,3.804162,3.885521,-1.246890,-1.537572,-3.411194,-6.678731,2.424965,2.053097,-0.509067,-9.807718,1.169434,-7.762866,5.912820,-5.125199,-6.034293,4.184926,9.514499,3.242898,2.059584,6.022730,-4.428720,-8.452048,-5.010940,5.530843,-8.382310,5.210248,-0.990215,-9.707637,-4.006240,-9.017832]], dtype = "float64")#candidate|17848|(1, 660)|const|float64
call_17847 = func_5513_call(relay.reshape(const_17848.astype('float64'), [660,]))
call_17849 = func_5513_call(relay.reshape(const_17848.astype('float64'), [660,]))
func_9127_call = mod.get_global_var('func_9127')
func_9128_call = mutated_mod.get_global_var('func_9128')
call_17875 = func_9127_call()
call_17876 = func_9127_call()
bop_17884 = relay.logical_and(const_17848.astype('bool'), relay.reshape(call_17847.astype('bool'), relay.shape_of(const_17848))) # shape=(1, 660)
bop_17887 = relay.logical_and(const_17848.astype('bool'), relay.reshape(call_17849.astype('bool'), relay.shape_of(const_17848))) # shape=(1, 660)
output = relay.Tuple([call_17839,call_17875,bop_17884,])
output2 = relay.Tuple([call_17840,call_17876,bop_17887,])
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