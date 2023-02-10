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
var_57 = relay.var("var_57", dtype = "float64", shape = (10, 13, 7))#candidate|57|(10, 13, 7)|var|float64
var_58 = relay.var("var_58", dtype = "float64", shape = (10, 13, 7))#candidate|58|(10, 13, 7)|var|float64
bop_59 = relay.greater(var_57.astype('bool'), relay.reshape(var_58.astype('bool'), relay.shape_of(var_57))) # shape=(10, 13, 7)
output = relay.Tuple([bop_59,])
output2 = relay.Tuple([bop_59,])
func_67 = relay.Function([var_57,var_58,], output)
mod['func_67'] = func_67
mod = relay.transform.InferType()(mod)
mutated_mod['func_67'] = func_67
mutated_mod = relay.transform.InferType()(mutated_mod)
func_67_call = mutated_mod.get_global_var('func_67')
var_69 = relay.var("var_69", dtype = "float64", shape = (10, 13, 7))#candidate|69|(10, 13, 7)|var|float64
var_70 = relay.var("var_70", dtype = "float64", shape = (10, 13, 7))#candidate|70|(10, 13, 7)|var|float64
call_68 = func_67_call(var_69,var_70,)
output = call_68
func_71 = relay.Function([var_69,var_70,], output)
mutated_mod['func_71'] = func_71
mutated_mod = relay.transform.InferType()(mutated_mod)
var_185 = relay.var("var_185", dtype = "int64", shape = (4, 13, 11))#candidate|185|(4, 13, 11)|var|int64
var_186 = relay.var("var_186", dtype = "int64", shape = (4, 13, 11))#candidate|186|(4, 13, 11)|var|int64
bop_187 = relay.bitwise_xor(var_185.astype('int64'), relay.reshape(var_186.astype('int64'), relay.shape_of(var_185))) # shape=(4, 13, 11)
func_67_call = mod.get_global_var('func_67')
func_71_call = mutated_mod.get_global_var('func_71')
const_192 = relay.const([-6.394749,8.327457,-5.543298,-0.790615,2.623398,2.919884,7.531010,4.967688,-6.969119,4.104499,-1.179110,9.761528,6.155867,-9.804660,-9.212046,3.856843,-4.258950,-7.069380,-7.296764,1.341450,-8.777630,2.023806,6.195526,3.175559,7.153789,-4.266769,2.804210,-1.090653,-1.225037,5.304431,-7.848576,8.651776,-5.871123,5.902223,-5.923564,-7.070467,-6.646075,-8.914607,-0.177971,7.454013,-8.293912,0.491604,8.841600,7.536200,-2.085466,-0.055238,3.341128,-4.629121,-1.820635,-4.504725,-8.726572,7.825751,9.003621,-0.262562,9.197799,2.444999,8.011045,3.553661,-6.408915,-4.863799,5.193894,5.152430,2.129029,-4.907420,-3.835671,-3.057955,9.550405,0.613294,-4.675708,7.591525,2.663730,1.114834,0.841789,9.225461,5.954965,0.976771,-7.763926,0.545715,-5.612232,0.868471,9.811833,5.709934,4.383479,8.593365,1.454599,2.965793,4.583337,1.936140,5.606398,3.272869,5.555038,-0.565462,3.889060,1.049751,8.595368,-9.434116,-8.048398,-0.971698,-7.520867,0.360665,-9.557463,-7.560244,-3.570501,-4.640342,-6.420375,-8.696766,-7.026018,-4.530233,3.509797,2.128432,-5.120347,2.746947,-5.634729,9.489601,8.200404,-1.680527,3.433102,4.392663,-9.895943,-7.355497,1.637603,-7.491842,5.243710,-2.675439,-5.538915,1.972275,9.513986,-3.331357,-7.773849,6.923341,-1.925510,4.199538,-0.357410,5.894234,5.290019,-6.405604,9.609857,-2.525901,0.765068,9.309006,1.280549,-8.379153,4.232587,-5.490453,-8.124526,3.602988,3.524416,0.462629,5.903189,-4.794985,-9.724961,-9.643438,-6.068460,-4.352408,4.148725,1.842328,-0.195130,5.395006,-4.363698,-8.973540,4.346673,-9.161655,-3.410037,9.163188,-7.942211,-0.909553,-1.814677,-8.069633,-8.080004,-2.220212,-8.618547,-3.139477,5.129516,-3.451970,1.728015,6.527834,-4.739578,7.139676,4.923139,0.411471,-5.885065,-2.193125,-9.654206,7.141524,8.463882,-5.872007,-4.273891,-2.188425,-8.993569,3.790396,8.116471,4.713882,-0.075258,0.762248,9.902688,-7.021300,-2.390913,-5.652857,1.938144,6.544645,8.701293,8.555506,-2.285143,1.195103,-1.812439,-2.784114,7.586789,-7.538617,-3.472378,4.669928,-7.320688,-4.369997,-6.509164,2.255463,-8.217463,-5.871133,-6.532967,1.638682,-5.279422,8.710237,-8.405614,-1.143627,9.734977,0.369988,-4.470557,2.599965,-9.653167,-5.482139,-3.980821,2.159356,-8.835614,-0.375926,4.214471,-2.068904,2.387275,-5.828910,0.239876,7.553058,8.950880,6.575535,-9.732498,-3.603238,-2.424293,1.679636,-6.009964,2.521564,0.724527,0.734841,-9.008375,-1.165673,-2.347278,6.581647,6.001894,-6.637417,3.616043,4.793696,7.375794,5.603765,-0.104821,-9.416715,2.020266,3.683570,-2.446021,3.446961,-2.661218,-3.680938,-0.152772,7.895606,-3.962918,2.762448,0.233402,8.699114,-3.692801,-0.258733,-6.720877,-1.280745,-2.265090,-7.095112,-0.627921,2.267419,4.112197,2.590264,3.653477,8.175504,2.334837,-9.446217,2.779936,9.944609,-9.932316,-7.467254,-8.967996,-9.331194,0.206438,5.503109,-9.287405,-8.380417,-3.887143,9.224259,-4.699300,-8.680274,2.007449,-5.746069,-4.018546,3.285319,1.773547,1.349628,-9.402053,4.468000,-0.430125,5.342726,1.216845,-3.209050,-2.829503,-7.645823,8.803401,-8.239671,0.222620,6.879019,-4.645138,-7.243895,9.110982,9.297474,3.012163,1.569526,-8.612849,-3.016281,-1.556163,6.335189,-4.381313,-9.074406,-1.756052,2.618723,3.164816,-0.517473,8.745880,-8.421618,-6.164725,-1.188434,-0.601448,-2.391309,-1.568521,-7.035454,6.664200,8.928602,8.020185,4.394923,-2.369855,8.242137,-7.389666,0.583260,-0.470104,-9.534152,-8.017869,-0.635037,-7.505908,2.882173,2.470618,2.964384,9.490140,6.409330,-0.135244,6.845115,9.390106,6.906116,-1.356987,9.226467,-2.776174,4.024534,-6.043073,4.748281,-3.617261,8.876060,-5.211568,8.530345,-8.611755,7.566998,5.544277,7.084548,6.977704,-6.245860,-8.222742,9.107514,5.764729,7.279516,-2.440902,-1.531092,8.909492,0.953527,6.685035,0.403364,7.698656,-7.769433,1.006428,7.460693,-0.680869,-8.316273,6.034185,9.911966,2.061701,-5.794011,5.030474,-7.147929,-1.909739,0.940452,-7.418756,2.443497,8.611101,1.499423,-6.999072,2.222690,-1.969882,2.039024,7.926945,1.341454,2.406057,3.422079,-5.252673,3.266784,4.356498,-7.303731,8.692689,2.484618,-1.188324,-2.901924,1.230940,4.163555,6.236535,0.975265,-5.902466,8.315574,-7.020291,-5.890180,0.357146,8.799757,-0.762140,-1.144942,-2.765978,-2.154663,-7.295598,1.432830,9.799201,-1.284929,8.135980,3.542153,-2.130838,0.400322,0.229234,7.556782,1.241500,3.925435,2.081116,2.370261,0.038773,-0.914325,-6.741897,1.975211,-1.688702,1.833394,5.594368,3.350109,5.918890,-5.997440,2.425864,6.908234,4.022553,-2.678911,1.046075,-8.295188,-3.929103,-6.050424,-3.664443,-5.781753,3.588262,-2.168191,-3.764436,-7.107140,-6.677961,3.264948,9.075840,-5.915781,7.994684,1.475699,-9.870090,-5.930766,-2.572325,-2.566862,-0.132535,4.392024,-7.249865,6.833258,-9.391118,-0.748890,-9.285413,-0.119543,0.392677,-6.233760,9.129428,5.213081,0.556386,-8.593618,-7.022054,-7.913394,9.191275,6.960752,1.588893,7.573273,-8.741436,-8.299772,3.568384,-2.853712,8.196606,8.866626,-8.609706,0.263762,-3.270638,2.391761,-9.970077,-9.382940,-4.057761,5.202964,-0.815043,2.154217,-5.999571,0.110791,-8.409658,-4.099517,1.782244,-3.219102,-2.213728,0.588110,-8.899815,-2.392305,8.404959,-7.238368,-6.681607,1.296619,-8.201705,8.677491,3.229109,-6.332787,1.709360,3.162268,5.707099,-4.334624,4.118898,-7.790753,0.794966,-9.118678,8.580434,-0.565602,-7.402594,-0.147533,-2.986654,-3.017058,6.548684,6.582857,4.071289,8.079058,-8.904916,9.283786,0.679441,8.207811,-6.380420,0.651256,6.408110,8.690806,-5.035320,-5.993986,3.685918,-2.992317,-0.292938,6.357151,-8.131904,-8.799837,3.784422,9.847195,0.554364,6.459610,-4.664281,-0.966187,-5.065997,-0.231645,3.047297,1.672545,5.084427,3.186895,9.786396,0.229934,9.215142,-5.512505,0.597138,-0.600053,9.578540,-6.428683,-2.476780,4.658539,-6.118875,0.961497,-0.495103,2.171376,-3.554693,8.640683,6.318009,-6.293922,-4.031347,1.356824,-6.782256,3.694202,6.014230,2.652587,-9.751571,-7.469631,-8.875702,-2.734545,-0.197507,-7.451410,7.319025,-3.946053,0.358599,5.153853,-8.709195,8.963881,-1.319013,-2.737020,6.947720,-8.759773,-5.031703,-6.187115,-8.588027,4.443153,-6.851806,-4.188531,-8.839108,0.297151,2.288945,1.856321,7.404768,-9.536610,1.447551,-3.384699,-6.202649,-0.655024,9.598240,-5.682692,-9.110400,7.838142,6.025088,-3.915448,-4.573105,5.308170,-3.733043,5.754482,-2.409876,-7.163465,-2.332794,-9.063513,7.080997,4.426459,4.858646,-4.585297,-4.243691,5.164682,6.067670,0.232908,8.131693,9.200085,3.759561,1.800382,-0.427803,-5.694264,-5.799569,-7.035169,2.083976,3.357192,4.595404,-6.533092,4.748915,8.025299,1.864082,-0.437376,-4.576524,2.719431,3.997435,1.016915,-9.228740,-3.353537,-2.432999,-5.173453,7.382793,-2.755403,8.355545,-2.553856,-2.493686,-0.389587,-3.492479,-2.793108,-5.927493,-6.995553,9.119226,-8.227288,7.182633,-8.857089,1.354169,-2.805246,-5.221130,1.339613,1.511351,-7.886888,4.710469,-7.013947,-9.098524,-9.139764,-4.917811,-6.432477,-3.904994,5.811615,6.083819,3.572048,-3.516796,2.055011,2.103612,-0.191017,-9.275043,-9.438900,1.465101,2.638819,-7.303265,7.173287,9.753010,-5.174903,-4.986774,0.502620,1.625263,-2.995556,-6.246376,7.977101,-2.809049,7.605986,-5.967926,-9.814433,-1.920680,5.697462,-9.322369,8.017054,-4.467480,9.475199,2.501255,0.563848,-3.667579,-9.390460,4.299009,-8.001165,5.024733,7.648225,5.582213,-1.241508,5.896925,-5.375580,-1.429799,4.560885,5.449248,3.862625,-1.012665,-6.101410,-9.941385,4.133906,-4.224454,2.218740,1.520611,8.989706,-7.798540,-9.279726,-4.863670,9.411375,-0.499312,1.957555,-6.313359,-8.745479,0.527381,3.082349,0.134267,-7.840394,-4.981135,-0.494297,-3.444657,-4.661757,4.870041,-8.973812,-5.925999,-2.807449,9.911254,-8.639889,-9.715641,-0.110556,5.094373,1.107113,-3.042709,-9.208922,7.085251,9.567129,9.434591,-5.503476,-8.855850,-8.127821,1.378040,-8.203907,-9.467437,-5.653710,2.752133,3.959586,-3.795293,-5.534434,-7.385605,4.113551,3.613271,6.385653,-6.276675,6.855985,-1.172020,7.488676,-7.134685,-9.370903,0.583050,4.552150,4.725295,2.164250,1.114135,2.517380,5.936039,9.104194,3.247553,9.701376,-3.596241,-2.433685,-7.987394,-5.581296,3.123806,3.973467,7.621754,-4.775779,9.739754,-0.182806,6.438120,-2.689805,-8.443351,2.984383,-9.368376,-4.433450,8.517865,-9.990493,-9.140340,-2.576043,6.777764,-6.285301,-5.677464,-1.885206,3.156342,-3.225436,-5.196206,7.980385,-7.290693,-9.334184,1.363251,5.107804,0.428517,-8.668260,6.412561,-9.951686,-0.061015,1.070685,7.364500,7.527274,0.779605,8.189355,-7.165770,-7.604134,2.189238,-6.745254,7.036701,6.623401,7.458444,3.743840,-5.327216,5.450136,-9.865880,9.662283,-5.436837,-1.381440,-3.392374,1.286743,-6.871659,2.150880,-5.308145,-1.401280,0.472565,1.321851,9.339749,1.722272,-9.536961,1.248613,-3.197224,-2.779471,7.757322,-2.965591,-8.615286,-9.249210,3.211241,-6.804105,5.163453,-6.820285], dtype = "float64")#candidate|192|(910,)|const|float64
call_191 = relay.TupleGetItem(func_67_call(relay.reshape(const_192.astype('float64'), [10, 13, 7]), relay.reshape(const_192.astype('float64'), [10, 13, 7]), ), 0)
call_193 = relay.TupleGetItem(func_71_call(relay.reshape(const_192.astype('float64'), [10, 13, 7]), relay.reshape(const_192.astype('float64'), [10, 13, 7]), ), 0)
bop_216 = relay.logical_and(var_185.astype('bool'), relay.reshape(bop_187.astype('bool'), relay.shape_of(var_185))) # shape=(4, 13, 11)
bop_220 = relay.add(bop_187.astype('uint8'), relay.reshape(var_186.astype('uint8'), relay.shape_of(bop_187))) # shape=(4, 13, 11)
const_229 = relay.const([[[False,False,True,False,False,True,True,True,False,True,True],[True,False,True,False,False,True,False,True,True,False,False],[True,False,True,False,True,False,True,True,False,True,True],[True,True,True,True,False,False,True,True,True,False,True],[False,False,True,True,True,False,True,True,True,False,True],[False,False,False,False,True,True,True,False,False,False,True],[True,True,True,True,False,False,False,True,False,False,True],[True,False,False,False,True,True,True,True,False,True,True],[True,True,True,False,False,True,True,True,True,False,False],[False,False,True,True,False,False,True,True,True,False,True],[False,False,False,True,True,True,False,False,False,True,False],[True,True,False,True,False,True,False,True,False,False,True],[True,True,True,True,False,False,False,True,True,False,True]],[[True,False,True,False,False,True,True,True,True,False,False],[False,True,True,True,True,False,True,False,True,False,True],[False,True,False,True,False,True,False,False,False,False,False],[True,False,False,True,True,True,False,False,False,False,True],[True,False,True,True,True,True,False,False,False,False,True],[False,True,True,True,False,False,False,False,False,True,False],[True,False,False,False,True,False,False,True,True,True,True],[False,False,True,True,False,True,False,True,False,False,False],[False,False,False,False,False,False,False,True,True,True,False],[True,False,True,True,False,False,False,False,False,True,False],[True,False,True,False,False,False,False,True,False,False,True],[True,False,True,True,True,False,False,False,False,True,False],[True,False,True,False,True,True,False,False,True,True,False]],[[False,True,True,False,True,False,True,False,True,False,False],[False,True,True,False,False,False,True,True,False,False,False],[True,False,True,False,False,True,False,True,True,False,False],[False,True,False,False,False,False,True,False,False,True,True],[False,False,True,False,True,True,False,True,True,False,False],[False,False,False,True,False,False,False,True,False,True,False],[False,True,False,False,True,False,True,False,True,True,False],[True,True,True,False,True,True,True,True,True,True,False],[True,False,False,True,False,True,False,True,True,False,True],[True,True,False,True,False,False,True,True,True,False,False],[False,False,True,True,True,True,False,False,True,False,False],[True,True,False,True,True,False,False,False,True,False,True],[True,True,True,True,False,False,False,False,False,True,False]],[[True,False,True,False,True,True,True,True,True,True,False],[False,False,False,False,False,False,True,False,True,False,False],[False,True,False,False,False,True,False,True,True,True,True],[True,False,False,False,True,True,False,False,False,False,False],[False,False,True,False,True,True,True,True,True,True,False],[False,True,False,True,True,True,False,False,True,True,False],[True,False,False,True,False,False,True,False,True,True,False],[False,True,True,True,False,False,True,True,False,True,False],[True,False,True,False,False,True,True,False,False,True,False],[False,False,False,False,True,False,True,False,False,True,True],[True,True,True,True,False,False,True,True,False,False,True],[True,True,True,False,False,True,False,True,True,False,True],[True,True,False,True,False,False,True,False,True,False,True]]], dtype = "bool")#candidate|229|(4, 13, 11)|const|bool
bop_230 = relay.equal(bop_216.astype('bool'), relay.reshape(const_229.astype('bool'), relay.shape_of(bop_216))) # shape=(4, 13, 11)
output = relay.Tuple([call_191,const_192,bop_220,bop_230,])
output2 = relay.Tuple([call_193,const_192,bop_220,bop_230,])
func_241 = relay.Function([var_185,var_186,], output)
mod['func_241'] = func_241
mod = relay.transform.InferType()(mod)
var_242 = relay.var("var_242", dtype = "int64", shape = (4, 13, 11))#candidate|242|(4, 13, 11)|var|int64
var_243 = relay.var("var_243", dtype = "int64", shape = (4, 13, 11))#candidate|243|(4, 13, 11)|var|int64
output = func_241(var_242,var_243,)
func_244 = relay.Function([var_242,var_243,], output)
mutated_mod['func_244'] = func_244
mutated_mod = relay.transform.InferType()(mutated_mod)
const_333 = relay.const([[-3],[3]], dtype = "int32")#candidate|333|(2, 1)|const|int32
var_334 = relay.var("var_334", dtype = "int32", shape = (2, 3))#candidate|334|(2, 3)|var|int32
bop_335 = relay.add(const_333.astype('int32'), var_334.astype('int32')) # shape=(2, 3)
bop_339 = relay.left_shift(bop_335.astype('uint64'), const_333.astype('uint64')) # shape=(2, 3)
uop_354 = relay.acosh(var_334.astype('float32')) # shape=(2, 3)
var_356 = relay.var("var_356", dtype = "float32", shape = (2, 3))#candidate|356|(2, 3)|var|float32
bop_357 = relay.bitwise_or(uop_354.astype('uint64'), relay.reshape(var_356.astype('uint64'), relay.shape_of(uop_354))) # shape=(2, 3)
output = relay.Tuple([bop_339,bop_357,])
output2 = relay.Tuple([bop_339,bop_357,])
func_361 = relay.Function([var_334,var_356,], output)
mod['func_361'] = func_361
mod = relay.transform.InferType()(mod)
var_362 = relay.var("var_362", dtype = "int32", shape = (2, 3))#candidate|362|(2, 3)|var|int32
var_363 = relay.var("var_363", dtype = "float32", shape = (2, 3))#candidate|363|(2, 3)|var|float32
output = func_361(var_362,var_363,)
func_364 = relay.Function([var_362,var_363,], output)
mutated_mod['func_364'] = func_364
mutated_mod = relay.transform.InferType()(mutated_mod)
var_425 = relay.var("var_425", dtype = "float32", shape = (1, 12))#candidate|425|(1, 12)|var|float32
uop_426 = relay.sin(var_425.astype('float32')) # shape=(1, 12)
func_241_call = mod.get_global_var('func_241')
func_244_call = mutated_mod.get_global_var('func_244')
const_434 = relay.const([[3,6,-7,2,-8,-8,3,-2,2,-8,-9,-10,2,8,1,-3,-1,10,5,-7,2,-6,7,-1,8,-3,8,9,-9,5,2,-6,8,-10,-1,8,2,-8,1,-7,-1,-5,6,-10,2,6,-3,6,2,4,3,-7,-5,-6,-4,-7,-3,-5,6,-5,1,2,-6,5,4,-1,3,-2,3,-7,10,4,-6,-4,4,-3,-3,-7,5,8,1,-8,9,5,7,1,-5,3,-3,6,-8,-2,-5,-1,3,10,9,8,-8,-10,-4,6,-5,1,-8,7,5,9,10,-10,9,7,-10,6,8,6,-6,-3,1,-2,2,-7,9,-7,-7,-2,-9,-4,3,-8,5,-5,6,3,2,8,-5,3,10,4,8,-4,6,7,-8,-2,-1,-4,-5,-4,-4,-2,-7,2,-6,-7,-4,-3,-10,10,7,1,-1,5,-5,4,1,-8,-3,-8,-9,-5,-7,-5,-1,5,9,8,-4,-7,-3,-8,-8,-5,-7,2,-9,2,7,2,-4,-8,4,4,-10,-7,-9,1,1,10,10,-7,9,4,-5,-4,3,2,7,-7,7,1,-9,5,8,-10,-7,9,4,-4,2,-10,9,4,9,4,-3,2,-9,4,4,10,-10,-9,-4,7,-8,-1,-10,4,4,7,-7,1,-8,7,-10,-3,-6,9,3,6,-5,-10,-9,8,4,-3,4,9,1,-4,-1,-1,7,-6,8,-4,8,9,8,-1,1,-3,7,7,-8,-9,7,-5,-2,9,-3,-2,-8,-8,-8,-9,4,6,-6,9,5,2,-10,10,-9,1,-3,5,-5,8,-2,4,10,9,-4,-1,-6,5,-1,-7,6,-2,-2,8,10,3,-2,-9,-9,-2,7,-4,-8,9,5,-8,4,8,7,-3,2,8,-5,10,-3,6,6,3,10,1,-7,-4,9,9,2,2,10,-9,6,8,10,9,6,9,-3,-1,9,5,9,-3,-10,4,4,-6,3,-10,-7,-10,-1,9,9,-8,5,-6,1,-7,6,-2,7,3,-1,-4,1,1,2,10,10,-10,3,-9,-9,5,10,-8,-3,-6,3,3,5,-8,4,8,7,-8,6,-5,6,2,7,-2,2,-7,9,3,-5,-8,-10,-4,-6,5,2,1,-2,-5,-3,-9,4,-2,-7,-6,5,-5,2,-1,-2,4,-8,3,6,-7,4,7,-4,-10,3,5,-9,4,10,2,2,7,1,-8,-9,-9,2,10,5,9,5,10,6,7,-9,-9,9,3,-7,8,-7,-6,2,-7,5,7,-1,8,-10,3,-6,-2,-8,4,7,1,10,3,5,-1,7,2,6,7,-7,-7,5,-9,-6,10,6,7,1,9,1,6,-9,-10,-8,-1,9,9,7,1,-2,7,2,-5,-6,-9,-4,9,-3,8,-5,-8,7,10,-5,7,2,5,-10,9,7,-4,-2,6,-1,-9,-3,3,-4,4,-4,9,9,-3,6,-6,1,4,5,9,4,4,-4,-9,-10,10,-6,-1,4,-3,-8,-3,-8,-2,-9,-1]], dtype = "int64")#candidate|434|(1, 572)|const|int64
call_433 = relay.TupleGetItem(func_241_call(relay.reshape(const_434.astype('int64'), [4, 13, 11]), relay.reshape(const_434.astype('int64'), [4, 13, 11]), ), 0)
call_435 = relay.TupleGetItem(func_244_call(relay.reshape(const_434.astype('int64'), [4, 13, 11]), relay.reshape(const_434.astype('int64'), [4, 13, 11]), ), 0)
output = relay.Tuple([uop_426,call_433,const_434,])
output2 = relay.Tuple([uop_426,call_435,const_434,])
func_443 = relay.Function([var_425,], output)
mod['func_443'] = func_443
mod = relay.transform.InferType()(mod)
var_444 = relay.var("var_444", dtype = "float32", shape = (1, 12))#candidate|444|(1, 12)|var|float32
output = func_443(var_444)
func_445 = relay.Function([var_444], output)
mutated_mod['func_445'] = func_445
mutated_mod = relay.transform.InferType()(mutated_mod)
var_473 = relay.var("var_473", dtype = "float32", shape = (16, 4, 3))#candidate|473|(16, 4, 3)|var|float32
uop_474 = relay.tan(var_473.astype('float32')) # shape=(16, 4, 3)
var_482 = relay.var("var_482", dtype = "float32", shape = (16, 4, 3))#candidate|482|(16, 4, 3)|var|float32
bop_483 = relay.logical_xor(uop_474.astype('int8'), relay.reshape(var_482.astype('int8'), relay.shape_of(uop_474))) # shape=(16, 4, 3)
uop_492 = relay.asin(bop_483.astype('float64')) # shape=(16, 4, 3)
output = relay.Tuple([uop_492,])
output2 = relay.Tuple([uop_492,])
func_494 = relay.Function([var_473,var_482,], output)
mod['func_494'] = func_494
mod = relay.transform.InferType()(mod)
var_495 = relay.var("var_495", dtype = "float32", shape = (16, 4, 3))#candidate|495|(16, 4, 3)|var|float32
var_496 = relay.var("var_496", dtype = "float32", shape = (16, 4, 3))#candidate|496|(16, 4, 3)|var|float32
output = func_494(var_495,var_496,)
func_497 = relay.Function([var_495,var_496,], output)
mutated_mod['func_497'] = func_497
mutated_mod = relay.transform.InferType()(mutated_mod)
var_547 = relay.var("var_547", dtype = "int8", shape = (12, 1))#candidate|547|(12, 1)|var|int8
var_548 = relay.var("var_548", dtype = "int8", shape = (12, 12))#candidate|548|(12, 12)|var|int8
bop_549 = relay.bitwise_and(var_547.astype('int8'), var_548.astype('int8')) # shape=(12, 12)
const_555 = relay.const([[-10,-7,-6,-10,9,7,7,6,-9,-4,-6,8],[10,-3,-1,-8,-1,-7,-10,-1,8,-5,-9,4],[3,7,-8,-5,8,-7,6,-9,10,5,-4,1],[4,6,-6,1,8,-1,2,-7,3,-1,-9,9],[6,-1,10,-2,-7,8,-7,-5,9,-3,1,1],[-6,-6,-3,-6,6,-1,9,-1,-10,-10,-9,3],[-10,-9,10,6,4,-1,-6,8,-4,8,-2,10],[1,-2,4,10,-4,6,3,-10,-5,-6,7,2],[-4,-5,-3,-10,-4,7,-10,3,-6,-9,-2,-5],[-9,-9,-6,-8,-3,-7,-7,-3,1,-8,-6,4],[1,-2,2,1,10,-10,10,5,10,-4,-10,7],[-6,10,-7,6,8,2,-5,1,-2,2,-8,7]], dtype = "int8")#candidate|555|(12, 12)|const|int8
bop_556 = relay.greater(bop_549.astype('bool'), relay.reshape(const_555.astype('bool'), relay.shape_of(bop_549))) # shape=(12, 12)
func_494_call = mod.get_global_var('func_494')
func_497_call = mutated_mod.get_global_var('func_497')
var_567 = relay.var("var_567", dtype = "float32", shape = (96, 2))#candidate|567|(96, 2)|var|float32
call_566 = relay.TupleGetItem(func_494_call(relay.reshape(var_567.astype('float32'), [16, 4, 3]), relay.reshape(var_567.astype('float32'), [16, 4, 3]), ), 0)
call_568 = relay.TupleGetItem(func_497_call(relay.reshape(var_567.astype('float32'), [16, 4, 3]), relay.reshape(var_567.astype('float32'), [16, 4, 3]), ), 0)
const_569 = relay.const([[False,True,True,False,True,True,True,True,False,True,True,False],[True,True,False,True,True,False,True,True,True,False,False,False],[True,True,True,True,False,False,False,False,True,True,True,False],[True,False,True,True,True,True,True,False,False,True,False,True],[True,True,False,False,False,True,False,True,False,False,False,False],[False,False,False,True,False,True,True,True,True,False,False,False],[True,False,True,True,True,True,False,True,False,True,True,False],[False,False,True,False,False,False,False,False,False,False,True,False],[False,False,True,True,False,False,False,False,False,True,True,True],[False,False,False,False,True,False,True,True,False,False,True,False],[False,True,False,True,True,True,True,True,True,True,True,True],[True,True,True,True,False,True,False,False,True,True,False,False]], dtype = "bool")#candidate|569|(12, 12)|const|bool
bop_570 = relay.logical_or(bop_556.astype('bool'), relay.reshape(const_569.astype('bool'), relay.shape_of(bop_556))) # shape=(12, 12)
bop_578 = relay.less_equal(bop_556.astype('bool'), relay.reshape(const_569.astype('bool'), relay.shape_of(bop_556))) # shape=(12, 12)
var_583 = relay.var("var_583", dtype = "bool", shape = (12, 12))#candidate|583|(12, 12)|var|bool
bop_584 = relay.less(bop_578.astype('bool'), relay.reshape(var_583.astype('bool'), relay.shape_of(bop_578))) # shape=(12, 12)
bop_599 = relay.floor_mod(bop_570.astype('float32'), relay.reshape(bop_578.astype('float32'), relay.shape_of(bop_570))) # shape=(12, 12)
bop_620 = relay.logical_and(bop_570.astype('bool'), relay.reshape(bop_578.astype('bool'), relay.shape_of(bop_570))) # shape=(12, 12)
var_625 = relay.var("var_625", dtype = "bool", shape = (12, 12))#candidate|625|(12, 12)|var|bool
bop_626 = relay.bitwise_xor(bop_584.astype('int16'), relay.reshape(var_625.astype('int16'), relay.shape_of(bop_584))) # shape=(12, 12)
var_630 = relay.var("var_630", dtype = "bool", shape = (12, 12))#candidate|630|(12, 12)|var|bool
bop_631 = relay.logical_xor(bop_570.astype('uint8'), relay.reshape(var_630.astype('uint8'), relay.shape_of(bop_570))) # shape=(12, 12)
func_241_call = mod.get_global_var('func_241')
func_244_call = mutated_mod.get_global_var('func_244')
var_642 = relay.var("var_642", dtype = "int64", shape = (572,))#candidate|642|(572,)|var|int64
call_641 = relay.TupleGetItem(func_241_call(relay.reshape(var_642.astype('int64'), [4, 13, 11]), relay.reshape(var_642.astype('int64'), [4, 13, 11]), ), 3)
call_643 = relay.TupleGetItem(func_244_call(relay.reshape(var_642.astype('int64'), [4, 13, 11]), relay.reshape(var_642.astype('int64'), [4, 13, 11]), ), 3)
output = relay.Tuple([call_566,var_567,bop_599,bop_620,bop_626,bop_631,call_641,var_642,])
output2 = relay.Tuple([call_568,var_567,bop_599,bop_620,bop_626,bop_631,call_643,var_642,])
func_652 = relay.Function([var_547,var_548,var_567,var_583,var_625,var_630,var_642,], output)
mod['func_652'] = func_652
mod = relay.transform.InferType()(mod)
mutated_mod['func_652'] = func_652
mutated_mod = relay.transform.InferType()(mutated_mod)
func_652_call = mutated_mod.get_global_var('func_652')
var_654 = relay.var("var_654", dtype = "int8", shape = (12, 1))#candidate|654|(12, 1)|var|int8
var_655 = relay.var("var_655", dtype = "int8", shape = (12, 12))#candidate|655|(12, 12)|var|int8
var_656 = relay.var("var_656", dtype = "float32", shape = (96, 2))#candidate|656|(96, 2)|var|float32
var_657 = relay.var("var_657", dtype = "bool", shape = (12, 12))#candidate|657|(12, 12)|var|bool
var_658 = relay.var("var_658", dtype = "bool", shape = (12, 12))#candidate|658|(12, 12)|var|bool
var_659 = relay.var("var_659", dtype = "bool", shape = (12, 12))#candidate|659|(12, 12)|var|bool
var_660 = relay.var("var_660", dtype = "int64", shape = (572,))#candidate|660|(572,)|var|int64
call_653 = func_652_call(var_654,var_655,var_656,var_657,var_658,var_659,var_660,)
output = call_653
func_661 = relay.Function([var_654,var_655,var_656,var_657,var_658,var_659,var_660,], output)
mutated_mod['func_661'] = func_661
mutated_mod = relay.transform.InferType()(mutated_mod)
var_671 = relay.var("var_671", dtype = "uint16", shape = (9, 11, 16))#candidate|671|(9, 11, 16)|var|uint16
var_672 = relay.var("var_672", dtype = "uint16", shape = (9, 11, 16))#candidate|672|(9, 11, 16)|var|uint16
bop_673 = relay.less_equal(var_671.astype('bool'), relay.reshape(var_672.astype('bool'), relay.shape_of(var_671))) # shape=(9, 11, 16)
uop_676 = relay.erf(bop_673.astype('float32')) # shape=(9, 11, 16)
uop_682 = relay.log2(uop_676.astype('float64')) # shape=(9, 11, 16)
uop_688 = relay.tan(uop_682.astype('float64')) # shape=(9, 11, 16)
uop_692 = relay.cos(uop_682.astype('float32')) # shape=(9, 11, 16)
uop_704 = relay.log(uop_688.astype('float32')) # shape=(9, 11, 16)
func_361_call = mod.get_global_var('func_361')
func_364_call = mutated_mod.get_global_var('func_364')
var_712 = relay.var("var_712", dtype = "int32", shape = (6, 1))#candidate|712|(6, 1)|var|int32
call_711 = relay.TupleGetItem(func_361_call(relay.reshape(var_712.astype('int32'), [2, 3]), relay.reshape(var_712.astype('float32'), [2, 3]), ), 1)
call_713 = relay.TupleGetItem(func_364_call(relay.reshape(var_712.astype('int32'), [2, 3]), relay.reshape(var_712.astype('float32'), [2, 3]), ), 1)
func_361_call = mod.get_global_var('func_361')
func_364_call = mutated_mod.get_global_var('func_364')
call_718 = relay.TupleGetItem(func_361_call(relay.reshape(var_712.astype('int32'), [2, 3]), relay.reshape(var_712.astype('float32'), [2, 3]), ), 1)
call_719 = relay.TupleGetItem(func_364_call(relay.reshape(var_712.astype('int32'), [2, 3]), relay.reshape(var_712.astype('float32'), [2, 3]), ), 1)
var_723 = relay.var("var_723", dtype = "float32", shape = (9, 11, 16))#candidate|723|(9, 11, 16)|var|float32
bop_724 = relay.less(uop_704.astype('bool'), relay.reshape(var_723.astype('bool'), relay.shape_of(uop_704))) # shape=(9, 11, 16)
output = relay.Tuple([uop_692,call_711,var_712,call_718,bop_724,])
output2 = relay.Tuple([uop_692,call_713,var_712,call_719,bop_724,])
func_727 = relay.Function([var_671,var_672,var_712,var_723,], output)
mod['func_727'] = func_727
mod = relay.transform.InferType()(mod)
mutated_mod['func_727'] = func_727
mutated_mod = relay.transform.InferType()(mutated_mod)
func_727_call = mutated_mod.get_global_var('func_727')
var_729 = relay.var("var_729", dtype = "uint16", shape = (9, 11, 16))#candidate|729|(9, 11, 16)|var|uint16
var_730 = relay.var("var_730", dtype = "uint16", shape = (9, 11, 16))#candidate|730|(9, 11, 16)|var|uint16
var_731 = relay.var("var_731", dtype = "int32", shape = (6, 1))#candidate|731|(6, 1)|var|int32
var_732 = relay.var("var_732", dtype = "float32", shape = (9, 11, 16))#candidate|732|(9, 11, 16)|var|float32
call_728 = func_727_call(var_729,var_730,var_731,var_732,)
output = call_728
func_733 = relay.Function([var_729,var_730,var_731,var_732,], output)
mutated_mod['func_733'] = func_733
mutated_mod = relay.transform.InferType()(mutated_mod)
var_741 = relay.var("var_741", dtype = "float64", shape = (9, 8))#candidate|741|(9, 8)|var|float64
uop_742 = relay.acosh(var_741.astype('float64')) # shape=(9, 8)
bop_750 = relay.mod(var_741.astype('float64'), relay.reshape(uop_742.astype('float64'), relay.shape_of(var_741))) # shape=(9, 8)
func_494_call = mod.get_global_var('func_494')
func_497_call = mutated_mod.get_global_var('func_497')
var_754 = relay.var("var_754", dtype = "float32", shape = (48, 4))#candidate|754|(48, 4)|var|float32
call_753 = relay.TupleGetItem(func_494_call(relay.reshape(var_754.astype('float32'), [16, 4, 3]), relay.reshape(var_754.astype('float32'), [16, 4, 3]), ), 0)
call_755 = relay.TupleGetItem(func_497_call(relay.reshape(var_754.astype('float32'), [16, 4, 3]), relay.reshape(var_754.astype('float32'), [16, 4, 3]), ), 0)
func_652_call = mod.get_global_var('func_652')
func_661_call = mutated_mod.get_global_var('func_661')
var_758 = relay.var("var_758", dtype = "int8", shape = (12,))#candidate|758|(12,)|var|int8
const_759 = relay.const([5,-2,3,-7,10,-2,3,-7,-6,8,-1,-2,-3,-9,-6,-6,-2,7,-10,2,-4,-5,2,-8,10,-6,10,8,-2,10,-9,10,-9,-7,-1,1,3,4,7,4,8,8,1,-3,5,4,4,-5,2,1,3,1,-7,-3,-9,-2,10,-6,6,-5,-7,-8,2,-3,9,8,-8,-10,-5,9,-2,10,-1,-3,10,-1,-7,-7,2,1,-5,-3,-7,-8,-8,6,-8,-3,7,-5,6,-3,2,-9,-1,6,-9,-9,-4,1,-3,-3,3,7,2,6,-5,10,-10,-4,9,-9,2,7,4,-9,4,8,-8,9,7,9,-6,-1,-7,-9,6,-7,5,-4,3,-6,5,-6,8,-6,-1,-7,3,-5,-3,-7,4,7], dtype = "int8")#candidate|759|(144,)|const|int8
const_760 = relay.const([10,-1,5,-2,-4,-6,-1,-9,9,3,-2,3,6,-9,7,4,3,4,5,-9,-8,-10,-5,-3,10,2,3,-1,9,-5,3,-4,5,6,-3,-6,1,-7,-4,3,-8,-8,-10,2,-9,6,8,-10,-7,2,-3,-10,-6,-3,9,6,8,-10,3,8,1,-1,3,2,8,-9,-4,-5,7,-1,-1,6,1,10,10,-3,1,9,-7,-5,5,-6,-4,-9,1,7,-7,3,-1,-3,10,-1,1,-6,-1,5,-6,6,6,-1,5,8,3,3,2,-4,-3,-3,4,-5,10,-2,-3,-6,-6,-5,-7,7,4,-3,-6,-9,1,-5,3,10,2,-2,4,5,7,1,2,-3,7,-3,2,-4,-7,-7,-9,-6,3,-7,7,5,1,-7,5,9,7,-7,-3,-3,-9,1,-9,10,-6,-9,6,-9,1,10,-8,-7,9,4,-6,4,-7,-3,3,-10,-10,-4,-4,-3,-8,10,2,-1,-3,6,-1,-5,7,-9,-3,1,-10,10,-7,7,-8,3,6,2,8,7,10,2,-9,-10,-1,-6,8,-5,8,2,10,-5,-9,6,-7,-1,-4,-4,5,5,-5,-9,6,-7,7,-10,5,3,-2,-5,1,-4,10,-10,-2,-3,8,7,5,6,-3,10,2,6,-4,3,-7,-2,6,2,-9,-2,-4,-8,-3,2,3,-4,7,8,-10,-9,-9,4,3,1,8,6,2,-2,10,-1,-5,-7,6,6,-8,3,-5,3,10,9,-9,9,-1,-6,-4,-4,-3,-5,-8,-3,5,-6,-2,8,6,9,2,-9,-10,6,4,-6,2,-4,-2,1,-4,8,7,-7,3,8,-5,-10,2,10,5,2,7,10,-2,7,-7,9,2,6,-1,3,-3,-2,1,10,-8,-5,-8,-4,-5,6,7,7,3,-7,10,-2,-4,-8,6,1,2,-1,-3,1,-8,-10,-4,6,-2,6,2,-7,-4,-1,4,-6,-10,-7,2,10,6,-6,-9,-10,5,5,5,4,-6,10,9,-8,-1,-1,-6,-6,3,5,-6,4,-2,-1,-10,4,1,10,8,-8,-5,-5,-2,-3,2,-8,4,2,-3,-1,-9,3,-9,-10,10,-9,1,-8,-4,9,-1,-4,8,-9,10,-6,-4,-9,-1,-1,10,-10,3,-3,10,-1,-10,-4,-3,-1,-4,2,-5,3,9,-7,-4,-10,-4,2,-9,-5,-9,7,-3,-5,-7,-4,-7,8,-1,-6,1,-1,-1,1,5,-8,6,-8,-4,-3,-1,3,4,-3,6,9,5,6,-1,5,4,4,-9,-6,-7,-5,-6,4,10,4,-7,5,10,-10,-8,-1,-8,-2,-7,-1,-7,-3,-5,-6,6,-2,2,-7,-9,-5,1,1,1,6,10,-2,-4,-4,2,5,-10,3,4,-4,6,-2,-7,8,-6,4,4,5,-1,-8,-8,10,-9,10,2,8,5,-7,-3,-10,1,-7,2,-4,2,5,-4,8,6,9,-3,-6,-7,3,-7,-10,-7,-3,8,1,9,10,10,9,7,4,4,-10], dtype = "int64")#candidate|760|(572,)|const|int64
call_757 = relay.TupleGetItem(func_652_call(relay.reshape(var_758.astype('int8'), [12, 1]), relay.reshape(const_759.astype('int8'), [12, 12]), relay.reshape(var_754.astype('float32'), [96, 2]), relay.reshape(const_759.astype('bool'), [12, 12]), relay.reshape(const_759.astype('bool'), [12, 12]), relay.reshape(const_759.astype('bool'), [12, 12]), relay.reshape(const_760.astype('int64'), [572,]), ), 5)
call_761 = relay.TupleGetItem(func_661_call(relay.reshape(var_758.astype('int8'), [12, 1]), relay.reshape(const_759.astype('int8'), [12, 12]), relay.reshape(var_754.astype('float32'), [96, 2]), relay.reshape(const_759.astype('bool'), [12, 12]), relay.reshape(const_759.astype('bool'), [12, 12]), relay.reshape(const_759.astype('bool'), [12, 12]), relay.reshape(const_760.astype('int64'), [572,]), ), 5)
func_361_call = mod.get_global_var('func_361')
func_364_call = mutated_mod.get_global_var('func_364')
var_763 = relay.var("var_763", dtype = "int32", shape = (6,))#candidate|763|(6,)|var|int32
call_762 = relay.TupleGetItem(func_361_call(relay.reshape(var_763.astype('int32'), [2, 3]), relay.reshape(var_763.astype('float32'), [2, 3]), ), 1)
call_764 = relay.TupleGetItem(func_364_call(relay.reshape(var_763.astype('int32'), [2, 3]), relay.reshape(var_763.astype('float32'), [2, 3]), ), 1)
func_443_call = mod.get_global_var('func_443')
func_445_call = mutated_mod.get_global_var('func_445')
call_766 = relay.TupleGetItem(func_443_call(relay.reshape(var_758.astype('float32'), [1, 12])), 1)
call_767 = relay.TupleGetItem(func_445_call(relay.reshape(var_758.astype('float32'), [1, 12])), 1)
bop_768 = relay.bitwise_or(bop_750.astype('int8'), relay.reshape(uop_742.astype('int8'), relay.shape_of(bop_750))) # shape=(9, 8)
var_779 = relay.var("var_779", dtype = "float64", shape = (9, 8))#candidate|779|(9, 8)|var|float64
bop_780 = relay.maximum(bop_750.astype('uint16'), relay.reshape(var_779.astype('uint16'), relay.shape_of(bop_750))) # shape=(9, 8)
bop_783 = relay.minimum(uop_742.astype('int64'), relay.reshape(var_779.astype('int64'), relay.shape_of(uop_742))) # shape=(9, 8)
output = relay.Tuple([call_753,var_754,call_757,var_758,const_759,const_760,call_762,var_763,call_766,bop_768,bop_780,bop_783,])
output2 = relay.Tuple([call_755,var_754,call_761,var_758,const_759,const_760,call_764,var_763,call_767,bop_768,bop_780,bop_783,])
func_787 = relay.Function([var_741,var_754,var_758,var_763,var_779,], output)
mod['func_787'] = func_787
mod = relay.transform.InferType()(mod)
mutated_mod['func_787'] = func_787
mutated_mod = relay.transform.InferType()(mutated_mod)
func_787_call = mutated_mod.get_global_var('func_787')
var_789 = relay.var("var_789", dtype = "float64", shape = (9, 8))#candidate|789|(9, 8)|var|float64
var_790 = relay.var("var_790", dtype = "float32", shape = (48, 4))#candidate|790|(48, 4)|var|float32
var_791 = relay.var("var_791", dtype = "int8", shape = (12,))#candidate|791|(12,)|var|int8
var_792 = relay.var("var_792", dtype = "int32", shape = (6,))#candidate|792|(6,)|var|int32
var_793 = relay.var("var_793", dtype = "float64", shape = (9, 8))#candidate|793|(9, 8)|var|float64
call_788 = func_787_call(var_789,var_790,var_791,var_792,var_793,)
output = call_788
func_794 = relay.Function([var_789,var_790,var_791,var_792,var_793,], output)
mutated_mod['func_794'] = func_794
mutated_mod = relay.transform.InferType()(mutated_mod)
var_802 = relay.var("var_802", dtype = "int8", shape = (7, 1, 16))#candidate|802|(7, 1, 16)|var|int8
var_803 = relay.var("var_803", dtype = "int8", shape = (7, 6, 16))#candidate|803|(7, 6, 16)|var|int8
bop_804 = relay.multiply(var_802.astype('int8'), var_803.astype('int8')) # shape=(7, 6, 16)
bop_809 = relay.logical_or(var_802.astype('bool'), bop_804.astype('bool')) # shape=(7, 6, 16)
output = relay.Tuple([bop_809,])
output2 = relay.Tuple([bop_809,])
func_818 = relay.Function([var_802,var_803,], output)
mod['func_818'] = func_818
mod = relay.transform.InferType()(mod)
mutated_mod['func_818'] = func_818
mutated_mod = relay.transform.InferType()(mutated_mod)
func_818_call = mutated_mod.get_global_var('func_818')
var_820 = relay.var("var_820", dtype = "int8", shape = (7, 1, 16))#candidate|820|(7, 1, 16)|var|int8
var_821 = relay.var("var_821", dtype = "int8", shape = (7, 6, 16))#candidate|821|(7, 6, 16)|var|int8
call_819 = func_818_call(var_820,var_821,)
output = call_819
func_822 = relay.Function([var_820,var_821,], output)
mutated_mod['func_822'] = func_822
mutated_mod = relay.transform.InferType()(mutated_mod)
var_824 = relay.var("var_824", dtype = "int32", shape = ())#candidate|824|()|var|int32
var_825 = relay.var("var_825", dtype = "int32", shape = (1, 12))#candidate|825|(1, 12)|var|int32
bop_826 = relay.logical_xor(var_824.astype('int32'), var_825.astype('int32')) # shape=(1, 12)
output = relay.Tuple([bop_826,])
output2 = relay.Tuple([bop_826,])
func_830 = relay.Function([var_824,var_825,], output)
mod['func_830'] = func_830
mod = relay.transform.InferType()(mod)
var_831 = relay.var("var_831", dtype = "int32", shape = ())#candidate|831|()|var|int32
var_832 = relay.var("var_832", dtype = "int32", shape = (1, 12))#candidate|832|(1, 12)|var|int32
output = func_830(var_831,var_832,)
func_833 = relay.Function([var_831,var_832,], output)
mutated_mod['func_833'] = func_833
mutated_mod = relay.transform.InferType()(mutated_mod)
var_840 = relay.var("var_840", dtype = "float64", shape = (9, 8, 11))#candidate|840|(9, 8, 11)|var|float64
uop_841 = relay.atan(var_840.astype('float64')) # shape=(9, 8, 11)
output = uop_841
output2 = uop_841
func_855 = relay.Function([var_840,], output)
mod['func_855'] = func_855
mod = relay.transform.InferType()(mod)
var_856 = relay.var("var_856", dtype = "float64", shape = (9, 8, 11))#candidate|856|(9, 8, 11)|var|float64
output = func_855(var_856)
func_857 = relay.Function([var_856], output)
mutated_mod['func_857'] = func_857
mutated_mod = relay.transform.InferType()(mutated_mod)
const_866 = relay.const([[[5.669570],[0.055549],[1.846133],[-2.592013],[-9.320224],[6.586416],[-1.168737],[8.644392]],[[-9.442595],[3.630639],[-4.800008],[6.988262],[2.986795],[0.444842],[4.990199],[-2.654372]]], dtype = "float32")#candidate|866|(2, 8, 1)|const|float32
uop_867 = relay.sin(const_866.astype('float32')) # shape=(2, 8, 1)
uop_874 = relay.erf(uop_867.astype('float32')) # shape=(2, 8, 1)
var_878 = relay.var("var_878", dtype = "float32", shape = (2, 8, 9))#candidate|878|(2, 8, 9)|var|float32
bop_879 = relay.right_shift(uop_867.astype('uint8'), var_878.astype('uint8')) # shape=(2, 8, 9)
func_830_call = mod.get_global_var('func_830')
func_833_call = mutated_mod.get_global_var('func_833')
const_884 = relay.const(4, dtype = "int32")#candidate|884|()|const|int32
const_885 = relay.const([[10,4],[-10,-9],[7,1],[9,-7],[5,-10],[-7,-8]], dtype = "int32")#candidate|885|(6, 2)|const|int32
call_883 = relay.TupleGetItem(func_830_call(relay.reshape(const_884.astype('int32'), []), relay.reshape(const_885.astype('int32'), [1, 12]), ), 0)
call_886 = relay.TupleGetItem(func_833_call(relay.reshape(const_884.astype('int32'), []), relay.reshape(const_885.astype('int32'), [1, 12]), ), 0)
bop_893 = relay.bitwise_xor(bop_879.astype('int8'), uop_867.astype('int8')) # shape=(2, 8, 9)
bop_896 = relay.mod(uop_874.astype('float64'), bop_893.astype('float64')) # shape=(2, 8, 9)
output = relay.Tuple([call_883,const_884,const_885,bop_896,])
output2 = relay.Tuple([call_886,const_884,const_885,bop_896,])
func_902 = relay.Function([var_878,], output)
mod['func_902'] = func_902
mod = relay.transform.InferType()(mod)
mutated_mod['func_902'] = func_902
mutated_mod = relay.transform.InferType()(mutated_mod)
var_903 = relay.var("var_903", dtype = "float32", shape = (2, 8, 9))#candidate|903|(2, 8, 9)|var|float32
func_902_call = mutated_mod.get_global_var('func_902')
call_904 = func_902_call(var_903)
output = call_904
func_905 = relay.Function([var_903], output)
mutated_mod['func_905'] = func_905
mutated_mod = relay.transform.InferType()(mutated_mod)
const_927 = relay.const([[0.894831,-3.706838,5.449071,-8.334702,5.544696,-1.461007,2.617004,-1.965536,-5.054018,3.312902,-5.329757,0.938055,-6.646990,-7.654452],[8.689301,-5.393287,2.272620,6.097475,7.468641,9.893113,-9.859693,5.804879,-5.878217,-6.896792,0.385263,7.449652,2.948880,9.639002],[5.805811,7.026168,1.348730,4.078768,-8.734376,-6.829322,-6.935167,4.332787,7.262731,-8.853830,-0.519539,-3.965305,5.483942,3.705738],[-1.105805,-1.285569,-8.261367,-7.316964,0.969946,-3.519145,-0.924376,-7.938966,-5.777218,2.047434,-6.908303,-4.267119,-4.757739,-8.975652],[-5.703569,1.303371,3.324114,-1.162256,4.758741,5.304168,-1.791421,-2.587729,-6.340627,1.020955,-7.177265,-0.359947,-3.230602,0.836332],[-1.894360,-8.848092,8.175578,9.279758,7.478799,-3.313747,8.652510,-2.821905,-2.606424,-3.211118,9.726565,1.680878,1.662014,-7.400227],[-1.678307,-9.239619,6.372458,7.094619,-8.554179,6.457989,-8.596922,-1.848178,-3.401926,6.204259,-5.714564,4.644181,9.573091,6.325502],[3.071226,5.370084,-0.775336,-6.087850,-0.439455,-3.170235,-4.349867,1.526651,-8.899310,-7.579099,-9.532616,7.178797,-7.441582,-3.985203],[5.695319,-3.191721,3.320888,8.216476,7.985147,1.241626,1.872930,6.143301,-9.932257,7.270715,-9.334569,8.108525,3.841226,2.107121],[6.794484,-0.138507,-0.182830,6.112891,-6.049805,1.288082,-5.799425,5.119472,-5.988720,-5.253359,1.881039,8.203517,5.766465,-7.002871],[5.548823,-1.570188,2.973099,-3.652677,9.584904,9.556206,-3.979515,-2.546029,-1.834137,-5.183081,-6.743871,0.215663,-8.036482,6.839869]], dtype = "float32")#candidate|927|(11, 14)|const|float32
uop_928 = relay.log10(const_927.astype('float32')) # shape=(11, 14)
func_787_call = mod.get_global_var('func_787')
func_794_call = mutated_mod.get_global_var('func_794')
var_931 = relay.var("var_931", dtype = "float64", shape = (72,))#candidate|931|(72,)|var|float64
var_932 = relay.var("var_932", dtype = "float32", shape = (192,))#candidate|932|(192,)|var|float32
const_933 = relay.const([-7,-2,3,1,-10,4,8,6,9,-3,1,-8], dtype = "int8")#candidate|933|(12,)|const|int8
var_934 = relay.var("var_934", dtype = "int32", shape = (6,))#candidate|934|(6,)|var|int32
call_930 = relay.TupleGetItem(func_787_call(relay.reshape(var_931.astype('float64'), [9, 8]), relay.reshape(var_932.astype('float32'), [48, 4]), relay.reshape(const_933.astype('int8'), [12,]), relay.reshape(var_934.astype('int32'), [6,]), relay.reshape(var_931.astype('float64'), [9, 8]), ), 4)
call_935 = relay.TupleGetItem(func_794_call(relay.reshape(var_931.astype('float64'), [9, 8]), relay.reshape(var_932.astype('float32'), [48, 4]), relay.reshape(const_933.astype('int8'), [12,]), relay.reshape(var_934.astype('int32'), [6,]), relay.reshape(var_931.astype('float64'), [9, 8]), ), 4)
uop_937 = relay.log(uop_928.astype('float32')) # shape=(11, 14)
uop_941 = relay.atanh(uop_937.astype('float32')) # shape=(11, 14)
bop_959 = relay.floor_divide(uop_941.astype('float64'), relay.reshape(uop_928.astype('float64'), relay.shape_of(uop_941))) # shape=(11, 14)
func_727_call = mod.get_global_var('func_727')
func_733_call = mutated_mod.get_global_var('func_733')
const_968 = relay.const([[-3,-8],[-9,7],[4,1],[9,9],[-8,1],[-1,8],[9,1],[4,-7],[-6,-1],[-2,4],[-8,-9],[-5,-3],[9,3],[-8,5],[-7,-1],[8,7],[-2,10],[9,-1],[-6,-6],[-9,6],[-4,8],[-7,7],[9,-4],[7,2],[2,-5],[-5,9],[-9,2],[3,-5],[-6,9],[6,-9],[6,-4],[7,-4],[10,-9],[8,-6],[-1,4],[-8,-3],[2,10],[-9,-4],[4,-8],[8,7],[3,-4],[-4,-7],[-1,-10],[-10,3],[-6,1],[-4,10],[3,-9],[4,-10],[2,8],[-4,-10],[4,4],[-7,-7],[2,10],[-10,-3],[2,5],[-8,1],[-9,-7],[-3,-10],[8,-2],[-4,10],[4,-1],[-2,-9],[-6,10],[-1,3],[1,-7],[-8,-1],[2,3],[-2,-4],[-5,-3],[1,3],[6,1],[-4,9],[6,-10],[-2,3],[4,-4],[1,5],[2,4],[9,3],[9,-5],[10,-8],[2,3],[9,-10],[1,1],[1,-7],[5,-10],[-7,10],[-9,-8],[-10,7],[4,-6],[5,-6],[-1,6],[7,6],[-5,4],[9,7],[-8,-7],[3,10],[4,-2],[5,5],[7,9],[7,5],[4,-3],[3,1],[8,2],[3,-1],[-10,-8],[-1,3],[6,-3],[9,-2],[-7,-6],[-6,-4],[4,-8],[-9,-6],[1,6],[-6,-3],[2,5],[4,-4],[4,-1],[-4,7],[-2,7],[-3,2],[6,8],[-8,-4],[7,-2],[7,-3],[-4,3],[1,10],[-2,-7],[-9,-1],[-7,-8],[-1,5],[-2,-10],[3,-1],[-8,-6],[2,-4],[-2,-1],[10,-7],[2,1],[-8,1],[7,8],[-5,-8],[-3,5],[2,-2],[-2,7],[-7,-8],[8,4],[8,-5],[-7,5],[-5,-8],[-9,4],[2,-6],[-8,-6],[-5,-6],[4,10],[-7,8],[-1,-7],[-6,-3],[-8,6],[2,8],[-5,7],[-7,-7],[-4,4],[4,-5],[-5,-1],[9,-7],[2,1],[-7,6],[-3,6],[-9,-2],[9,-4],[8,-10],[-6,-5],[9,-6],[-7,-7],[-5,-10],[7,10],[-8,8],[-8,2],[4,8],[-5,6],[5,4],[-10,8],[-3,6],[4,-3],[6,-10],[4,10],[-4,-9],[-1,10],[9,-8],[5,4],[4,-8],[-4,9],[-4,6],[-8,7],[5,10],[1,4],[4,-8],[-9,8],[10,10],[-7,-6],[7,-9],[8,-9],[-2,-1],[-9,-2],[8,-3],[-10,-8],[7,6],[8,-6],[-2,4],[-1,-3],[1,7],[-10,4],[-8,-4],[-8,-3],[-4,-8],[9,-9],[-1,10],[-5,-4],[-8,8],[-9,-7],[1,-7],[8,3],[-9,4],[7,8],[6,7],[-3,-10],[3,3],[2,10],[3,-2],[9,1],[-6,-5],[-10,-8],[-3,10],[9,8],[-7,3],[1,5],[-3,5],[10,-10],[-2,-9],[8,-1],[-6,8],[8,-7],[-9,8],[-6,-5],[8,10],[7,6],[5,-4],[-7,-5],[-5,-8],[10,-9],[8,10],[8,2],[5,-6],[4,-4],[1,3],[4,-1],[-10,1],[6,9],[-7,9],[-4,3],[-5,-4],[5,7],[9,-4],[-7,-1],[5,6],[-5,-7],[-4,10],[-7,1],[-4,-10],[-4,-4],[8,-7],[4,1],[3,-3],[2,1],[-6,6],[-5,-5],[2,9],[6,-1],[1,9],[-8,-7],[8,7],[9,6],[8,9],[10,2],[1,10],[-5,4],[-10,3],[5,-9],[9,-3],[-4,4],[-2,5],[6,4],[9,7],[-6,9],[-5,-4],[-4,3],[8,2],[4,4],[-8,8],[6,-3],[5,-6],[10,6],[2,6],[-2,-9],[-5,-4],[6,7],[-7,-8],[-6,-8],[1,-1],[-3,-8],[7,-1],[5,9],[1,-10],[-1,1],[-3,-3],[-1,9],[6,5],[-7,4],[-9,2],[5,5],[-5,4],[-4,-6],[-5,-6],[-1,-10],[-7,8],[-10,-7],[-7,4],[-2,-9],[3,-5],[-10,6],[1,4],[8,-10],[-9,-10],[-1,9],[-8,2],[-1,7],[8,-3],[-2,-8],[10,6],[7,3],[-6,-4],[-4,7],[-5,1],[3,10],[-1,-2],[9,-7],[-1,6],[-1,-4],[4,4],[-1,9],[1,2],[2,-7],[-4,-6],[-6,-3],[-6,-5],[-10,1],[-6,-3],[7,5],[-5,-4],[-10,8],[8,10],[-6,1],[3,-8],[4,8],[-6,9],[4,1],[-2,6],[6,7],[-9,7],[9,7],[5,-5],[-7,-1],[4,4],[3,-9],[2,8],[-9,-1],[-1,6],[-10,1],[10,-6],[-9,-10],[4,-2],[1,3],[-7,3],[-5,6],[5,-5],[-7,-2],[-2,-4],[3,3],[7,-8],[7,4],[-4,9],[1,7],[-7,5],[-9,-3],[-5,-10],[-10,-10],[8,-7],[-9,6],[10,-5],[1,1],[4,-4],[-2,4],[3,-8],[6,-9],[-5,-9],[1,-10],[-1,2],[-8,-1],[-8,7],[-1,7],[-4,-6],[10,9],[6,-10],[3,6],[6,7],[-10,6],[5,-7],[-4,-10],[-7,3],[4,3],[-7,-4],[4,6],[9,-3],[4,-6],[3,-5],[6,-7],[2,2],[-1,-3],[-9,-1],[-9,4],[10,7],[-9,3],[6,10],[1,-7],[-9,-10],[-6,-1],[2,-10],[8,-8],[-5,4],[-8,7],[5,6],[-7,1],[-1,-8],[-4,2],[10,-6],[9,-1],[-9,-8],[2,-10],[5,4],[-6,9],[6,6],[6,-5],[-3,-7],[8,-7],[-2,3],[4,-10],[-9,2],[9,-5],[-9,-4],[-9,3],[3,6],[-7,8],[10,3],[5,5],[-10,-9],[8,-4],[5,4],[-1,7],[9,5],[6,5],[-7,4],[-2,5],[-3,10],[10,-10],[4,-2],[3,10],[3,-7],[-1,9],[-1,-9],[1,4],[10,-7],[10,7],[1,10],[9,5],[-5,2],[-5,5],[5,8],[10,10],[-5,-2],[4,10],[9,9],[-10,-9],[-9,3],[4,-9],[-4,7],[-10,-3],[3,10],[5,5],[10,1],[10,-4],[-2,2],[-7,-9],[9,-4],[-2,-4],[2,-4],[-5,-2],[-3,-4],[-4,-2],[9,6],[-7,6],[8,7],[-7,-7],[-6,-10],[10,-8],[7,-5],[-4,1],[3,9],[2,-7],[7,10],[-6,7],[-8,7],[-10,-10],[-6,-10],[-2,-4],[-8,-4],[-8,1],[-7,-1],[10,7],[4,-5],[-8,1],[-7,-6],[6,3],[-3,-4],[-8,-8],[8,7],[7,5],[7,-6],[-2,-1],[-10,10],[-5,9],[8,10],[-10,10],[-8,-5],[3,6],[9,-7],[9,-2],[4,-6],[-4,8],[-8,-7],[-7,-6],[8,-1],[1,-2],[-1,3],[-6,-10],[6,-1],[9,-2],[8,-7],[10,3],[7,8],[-3,-10],[1,4],[-10,-10],[-8,-7],[-10,8],[-2,-4],[-10,-10],[-1,9],[-3,8],[-4,-3],[-7,-1],[4,6],[8,-4],[10,-5],[-5,1],[7,1],[9,-4],[-4,-7],[10,7],[-5,9],[-8,-6],[-3,1],[7,-5],[10,-2],[7,-1],[-3,-1],[-6,6],[-3,7],[-8,-10],[-5,8],[-2,6],[-10,10],[6,-6],[2,1],[-8,-9],[2,10],[10,-10],[10,-7],[-10,4],[-3,7],[8,6],[3,-7],[-4,9],[4,-5],[-1,1],[2,6],[3,4],[-4,3],[4,-1],[-9,-7],[-5,-9],[10,-7],[-9,-6],[6,-3],[-2,5],[7,10],[-9,2],[9,1],[10,5],[1,2],[-10,-1],[10,7],[2,-10],[2,-4],[-10,9],[8,5],[7,7],[-2,10],[-7,10],[6,1],[-1,-10],[-2,-4],[-1,-2],[-10,-6],[-7,-6],[1,-2],[-8,8],[-4,-3],[4,8],[8,-8],[2,9],[-9,8],[5,8],[4,6],[4,6],[-7,-3],[-2,6],[-2,10],[-6,-5],[-3,2],[10,6],[10,4],[8,-4],[-10,-5],[-7,-4],[-3,4],[-5,-4],[-3,-10],[6,5],[-10,10],[10,4],[-10,-4],[-9,-7],[-5,4],[9,2],[-6,-4],[-10,8],[1,-9],[-1,-3],[-3,-9],[7,5],[-8,-4],[8,-10],[-10,-5],[-5,-2],[1,7],[6,5],[7,2],[-2,6],[7,-9],[-7,9],[3,-10],[9,5],[3,3],[2,-1],[-4,10],[1,7],[7,-4],[5,4],[9,8],[-5,-7],[-3,-8],[-10,3],[-1,1],[-9,9],[-4,-10],[6,9],[-7,-3],[-10,-9],[-3,-7],[5,7],[2,-9],[6,-7],[-2,-1],[1,4],[-9,-10],[-7,1],[7,3],[7,-9],[7,2],[-7,4],[-7,8],[7,8],[10,2],[-7,7],[10,-8],[-9,-9],[-7,10],[1,-3],[-10,-8],[6,-2],[6,-4],[-6,8],[1,4],[7,6],[4,9],[-1,6],[5,8],[-2,6],[2,-10],[10,-4],[-6,-1],[-2,6],[-5,1],[-7,-5],[3,-4],[4,10],[1,-4],[-9,-2],[-9,-7],[-1,6],[8,-2],[-6,-7],[-10,1],[-4,-10],[4,6],[4,-10],[2,-5],[2,1],[-9,-5],[9,4],[-5,-3],[-8,4],[9,-5],[9,-10],[-5,-7],[6,-7],[8,2],[5,7],[10,1],[-7,2],[-6,-2],[4,-7],[5,-6],[-2,8],[6,-9],[5,-1],[8,-5],[-6,2],[5,6],[-6,-10],[8,3],[-10,2],[-8,-7],[7,-10],[-9,-1],[-1,8],[6,9],[10,2],[7,8],[6,-3],[10,-9],[-9,1],[3,6],[6,-2],[-9,-4],[-2,9],[-4,7],[8,6],[2,-3],[4,1],[-8,1],[-1,6],[-6,9]], dtype = "uint16")#candidate|968|(792, 2)|const|uint16
call_967 = relay.TupleGetItem(func_727_call(relay.reshape(const_968.astype('uint16'), [9, 11, 16]), relay.reshape(const_968.astype('uint16'), [9, 11, 16]), relay.reshape(var_934.astype('int32'), [6, 1]), relay.reshape(const_968.astype('float32'), [9, 11, 16]), ), 3)
call_969 = relay.TupleGetItem(func_733_call(relay.reshape(const_968.astype('uint16'), [9, 11, 16]), relay.reshape(const_968.astype('uint16'), [9, 11, 16]), relay.reshape(var_934.astype('int32'), [6, 1]), relay.reshape(const_968.astype('float32'), [9, 11, 16]), ), 3)
func_443_call = mod.get_global_var('func_443')
func_445_call = mutated_mod.get_global_var('func_445')
call_977 = relay.TupleGetItem(func_443_call(relay.reshape(const_933.astype('float32'), [1, 12])), 2)
call_978 = relay.TupleGetItem(func_445_call(relay.reshape(const_933.astype('float32'), [1, 12])), 2)
uop_981 = relay.asin(bop_959.astype('float64')) # shape=(11, 14)
uop_992 = relay.sinh(uop_981.astype('float64')) # shape=(11, 14)
output = relay.Tuple([call_930,var_931,var_932,const_933,var_934,call_967,const_968,call_977,uop_992,])
output2 = relay.Tuple([call_935,var_931,var_932,const_933,var_934,call_969,const_968,call_978,uop_992,])
func_996 = relay.Function([var_931,var_932,var_934,], output)
mod['func_996'] = func_996
mod = relay.transform.InferType()(mod)
var_997 = relay.var("var_997", dtype = "float64", shape = (72,))#candidate|997|(72,)|var|float64
var_998 = relay.var("var_998", dtype = "float32", shape = (192,))#candidate|998|(192,)|var|float32
var_999 = relay.var("var_999", dtype = "int32", shape = (6,))#candidate|999|(6,)|var|int32
output = func_996(var_997,var_998,var_999,)
func_1000 = relay.Function([var_997,var_998,var_999,], output)
mutated_mod['func_1000'] = func_1000
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1036 = relay.const([[8.629504,-5.721380,0.861108,-7.403760,-1.339360,7.682496,-5.067022,9.327669,3.395791,-4.662636,-9.724752,-0.812237,-5.775024,-5.408189,0.828931,1.354854],[1.014469,0.288457,-6.543061,6.599840,-1.727404,-2.742506,1.367470,-6.099840,-6.350342,-4.557218,2.311901,2.381756,-6.865277,7.881172,1.158208,-0.959934],[-4.476973,6.623914,-6.792907,-8.154756,-0.054668,7.279316,-8.065360,-2.992014,-3.045594,0.960324,8.541875,8.245184,-7.973278,3.571920,-5.521701,-4.031641],[7.453198,7.246906,-2.647704,-2.028845,-8.282103,1.469503,6.952906,-1.816312,-2.743174,6.287854,-9.404225,7.403726,9.214679,-7.741152,-0.862906,6.073642],[-8.984790,2.067781,-8.197792,8.179928,6.062638,6.247808,-1.402959,5.694930,5.055669,-8.964747,5.575325,-1.999658,-7.575296,-4.484573,-1.113121,-1.183820],[2.362307,8.537134,8.764242,-8.648460,3.120378,4.710232,6.159382,-0.412447,-9.430630,-8.072530,-5.878659,-6.275072,3.231099,-4.754945,-0.496874,8.054736],[-9.537537,6.962422,3.566047,6.513577,4.348693,8.925763,7.285556,4.888303,-2.379819,4.203954,7.230839,-7.338327,3.320099,-4.906316,5.280792,3.217259],[-7.682396,-3.030051,-1.878570,7.123520,-9.651784,0.717567,-7.183290,5.503858,-8.131680,-5.175313,3.029212,0.812232,2.277707,-9.678144,4.351083,-4.381425],[7.385743,8.650750,-9.326377,4.800399,4.955473,-2.833929,-9.125179,1.139605,0.602431,-4.788123,-6.396175,2.596326,5.884262,-0.877490,6.633700,-6.388757],[5.939391,-4.313964,-6.561595,3.894412,-2.709441,-5.054896,6.659179,-5.247731,5.774557,7.991086,-9.597972,0.909856,0.260923,-3.395420,-4.874052,7.094826],[-0.824615,-7.011178,9.169808,3.670954,6.266524,-0.428387,3.214070,8.695590,0.846303,-1.466713,2.905623,1.931211,8.346978,8.781545,8.698795,-2.765979],[1.525277,-4.818541,-7.186491,-1.749202,0.286842,-5.391896,9.731001,6.382823,7.446498,-1.238702,8.623969,3.942379,9.016309,8.262581,2.409621,5.787883],[-9.433849,9.791695,3.358901,-5.745274,2.983918,9.724603,3.774474,0.850827,-4.750078,2.660913,-3.505901,-7.775549,-5.234208,-4.752854,8.861040,-8.689325]], dtype = "float32")#candidate|1036|(13, 16)|const|float32
uop_1037 = relay.sinh(const_1036.astype('float32')) # shape=(13, 16)
var_1040 = relay.var("var_1040", dtype = "float32", shape = (13, 16))#candidate|1040|(13, 16)|var|float32
bop_1041 = relay.multiply(uop_1037.astype('float32'), relay.reshape(var_1040.astype('float32'), relay.shape_of(uop_1037))) # shape=(13, 16)
func_443_call = mod.get_global_var('func_443')
func_445_call = mutated_mod.get_global_var('func_445')
var_1045 = relay.var("var_1045", dtype = "float32", shape = (12,))#candidate|1045|(12,)|var|float32
call_1044 = relay.TupleGetItem(func_443_call(relay.reshape(var_1045.astype('float32'), [1, 12])), 2)
call_1046 = relay.TupleGetItem(func_445_call(relay.reshape(var_1045.astype('float32'), [1, 12])), 2)
bop_1051 = relay.minimum(bop_1041.astype('int8'), relay.reshape(const_1036.astype('int8'), relay.shape_of(bop_1041))) # shape=(13, 16)
output = relay.Tuple([call_1044,var_1045,bop_1051,])
output2 = relay.Tuple([call_1046,var_1045,bop_1051,])
func_1054 = relay.Function([var_1040,var_1045,], output)
mod['func_1054'] = func_1054
mod = relay.transform.InferType()(mod)
var_1055 = relay.var("var_1055", dtype = "float32", shape = (13, 16))#candidate|1055|(13, 16)|var|float32
var_1056 = relay.var("var_1056", dtype = "float32", shape = (12,))#candidate|1056|(12,)|var|float32
output = func_1054(var_1055,var_1056,)
func_1057 = relay.Function([var_1055,var_1056,], output)
mutated_mod['func_1057'] = func_1057
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1087 = relay.var("var_1087", dtype = "float32", shape = (1, 4, 16))#candidate|1087|(1, 4, 16)|var|float32
uop_1088 = relay.cos(var_1087.astype('float32')) # shape=(1, 4, 16)
bop_1091 = relay.equal(uop_1088.astype('bool'), relay.reshape(var_1087.astype('bool'), relay.shape_of(uop_1088))) # shape=(1, 4, 16)
bop_1094 = relay.less(uop_1088.astype('bool'), relay.reshape(var_1087.astype('bool'), relay.shape_of(uop_1088))) # shape=(1, 4, 16)
uop_1105 = relay.sin(bop_1091.astype('float32')) # shape=(1, 4, 16)
bop_1111 = relay.mod(uop_1105.astype('float64'), relay.reshape(uop_1088.astype('float64'), relay.shape_of(uop_1105))) # shape=(1, 4, 16)
bop_1116 = relay.less(bop_1111.astype('bool'), relay.reshape(var_1087.astype('bool'), relay.shape_of(bop_1111))) # shape=(1, 4, 16)
uop_1119 = relay.log2(bop_1116.astype('float32')) # shape=(1, 4, 16)
output = relay.Tuple([bop_1094,uop_1119,])
output2 = relay.Tuple([bop_1094,uop_1119,])
func_1121 = relay.Function([var_1087,], output)
mod['func_1121'] = func_1121
mod = relay.transform.InferType()(mod)
var_1122 = relay.var("var_1122", dtype = "float32", shape = (1, 4, 16))#candidate|1122|(1, 4, 16)|var|float32
output = func_1121(var_1122)
func_1123 = relay.Function([var_1122], output)
mutated_mod['func_1123'] = func_1123
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1139 = relay.const([-0.228684,3.065864,0.578840,-1.966761,-1.149132,-4.727297,-1.850624,7.402201,6.176038,-2.938437,1.651970,7.579108,-5.724594,-3.942185], dtype = "float64")#candidate|1139|(14,)|const|float64
var_1140 = relay.var("var_1140", dtype = "float64", shape = (14,))#candidate|1140|(14,)|var|float64
bop_1141 = relay.divide(const_1139.astype('float64'), relay.reshape(var_1140.astype('float64'), relay.shape_of(const_1139))) # shape=(14,)
func_1054_call = mod.get_global_var('func_1054')
func_1057_call = mutated_mod.get_global_var('func_1057')
var_1155 = relay.var("var_1155", dtype = "float32", shape = (208,))#candidate|1155|(208,)|var|float32
var_1156 = relay.var("var_1156", dtype = "float32", shape = (12,))#candidate|1156|(12,)|var|float32
call_1154 = relay.TupleGetItem(func_1054_call(relay.reshape(var_1155.astype('float32'), [13, 16]), relay.reshape(var_1156.astype('float32'), [12,]), ), 1)
call_1157 = relay.TupleGetItem(func_1057_call(relay.reshape(var_1155.astype('float32'), [13, 16]), relay.reshape(var_1156.astype('float32'), [12,]), ), 1)
func_443_call = mod.get_global_var('func_443')
func_445_call = mutated_mod.get_global_var('func_445')
call_1158 = relay.TupleGetItem(func_443_call(relay.reshape(call_1154.astype('float32'), [1, 12])), 2)
call_1159 = relay.TupleGetItem(func_445_call(relay.reshape(call_1154.astype('float32'), [1, 12])), 2)
func_652_call = mod.get_global_var('func_652')
func_661_call = mutated_mod.get_global_var('func_661')
const_1177 = relay.const([9,10,-3,-4,-9,8,4,-2,8,-5,5,6,10,-1,3,6,9,-6,6,-5,-1,-4,7,-9,-3,-3,6,-10,9,3,-9,-5,-6,1,1,8,-10,-6,-10,2,-1,2,6,-4,-3,-8,7,-1,4,-4,9,-8,4,-6,9,8,4,-7,-1,-7,9,7,9,-6,9,10,2,6,2,-2,3,10,9,-8,-8,5,10,-1,1,4,9,-9,-3,-9,-9,7,9,9,-5,-8,-4,10,6,-5,-1,6,-6,-1,10,-8,-10,-10,2,-9,-8,10,10,8,-3,3,-5,-4,-4,1,-6,4,2,5,5,1,-10,9,-10,8,-3,7,-1,-8,7,-10,8,9,10,-6,-7,-2,7,-7,-10,2,5,9,2,3], dtype = "int8")#candidate|1177|(144,)|const|int8
const_1178 = relay.const([[-0.415827,6.244824,-3.892114,7.594146,-5.966233,-1.656185,-1.202788,2.916358,-4.376120,3.189578,1.725809,-4.430805,9.866088,0.737649,-2.505663,-9.381722,-2.490029,5.003933,-8.580876,-2.332589,4.589582,-8.701275,4.921511,5.152128,9.856982,1.786335,7.549307,7.298951,5.671617,3.966784,8.187202,-6.232345,7.004119,-0.030651,-7.211269,-7.949352,-3.034474,9.114372,-6.176814,1.250680,6.196157,-1.669111,-1.859466,1.407532,-3.860521,6.791046,1.249309,-1.610641,7.491525,-9.937971,7.608030,7.337579,4.020026,1.723651,-3.192988,0.272308,6.244502,-5.126292,-5.109860,9.913311,-6.955458,-4.217161,7.199373,-6.546380,2.061291,-2.100676,4.821868,-9.009389,8.113762,3.344083,6.670115,5.089893,-4.175097,-6.133274,-0.961711,-9.799300,2.619193,-7.144338,4.022668,9.920761,8.773445,7.924312,-6.103125,3.355238,-4.345672,2.971484,4.872236,-2.602423,-3.240349,-2.894140,5.424705,-3.354705,8.345122,9.547018,4.815693,-6.255955],[-8.592811,-4.833053,9.028799,-2.008921,-0.353852,8.663056,3.447916,6.635246,5.374670,-2.055106,-2.149516,4.744872,4.490669,-3.046832,5.704553,9.782860,8.916920,-0.060845,3.029869,-9.647741,-5.038225,5.259942,7.176796,-7.700583,4.717500,9.027535,0.049631,3.322279,0.761020,6.178346,-6.457370,8.935265,-1.962863,2.625885,-1.901574,1.503539,7.079976,-8.800961,-2.458942,1.394912,6.063823,2.002975,4.436381,8.249183,7.483526,-2.498281,1.405270,4.156285,0.724640,3.084552,-4.297441,7.401276,-9.986268,-0.092455,-9.400391,-5.741928,0.072271,-8.516822,-4.541944,6.584677,-1.950870,-4.257047,0.166276,-3.216439,-1.719253,0.299363,-8.705382,-3.360202,-8.325966,6.825251,6.387044,8.425191,-6.439008,5.207121,2.726207,-5.809503,-4.540718,-4.756902,-3.320175,6.056446,-1.263062,2.629520,9.971742,9.169092,-2.878464,-9.632033,-3.988350,-6.922929,4.406875,-8.795516,5.211080,-5.186611,7.378972,3.678846,8.660450,-6.055364]], dtype = "float32")#candidate|1178|(2, 96)|const|float32
call_1176 = relay.TupleGetItem(func_652_call(relay.reshape(var_1156.astype('int8'), [12, 1]), relay.reshape(const_1177.astype('int8'), [12, 12]), relay.reshape(const_1178.astype('float32'), [96, 2]), relay.reshape(const_1177.astype('bool'), [12, 12]), relay.reshape(const_1177.astype('bool'), [12, 12]), relay.reshape(const_1177.astype('bool'), [12, 12]), relay.reshape(call_1158.astype('int64'), [572,]), ), 1)
call_1179 = relay.TupleGetItem(func_661_call(relay.reshape(var_1156.astype('int8'), [12, 1]), relay.reshape(const_1177.astype('int8'), [12, 12]), relay.reshape(const_1178.astype('float32'), [96, 2]), relay.reshape(const_1177.astype('bool'), [12, 12]), relay.reshape(const_1177.astype('bool'), [12, 12]), relay.reshape(const_1177.astype('bool'), [12, 12]), relay.reshape(call_1158.astype('int64'), [572,]), ), 1)
output = relay.Tuple([bop_1141,call_1154,var_1155,var_1156,call_1158,call_1176,const_1177,const_1178,])
output2 = relay.Tuple([bop_1141,call_1157,var_1155,var_1156,call_1159,call_1179,const_1177,const_1178,])
func_1180 = relay.Function([var_1140,var_1155,var_1156,], output)
mod['func_1180'] = func_1180
mod = relay.transform.InferType()(mod)
mutated_mod['func_1180'] = func_1180
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1180_call = mutated_mod.get_global_var('func_1180')
var_1182 = relay.var("var_1182", dtype = "float64", shape = (14,))#candidate|1182|(14,)|var|float64
var_1183 = relay.var("var_1183", dtype = "float32", shape = (208,))#candidate|1183|(208,)|var|float32
var_1184 = relay.var("var_1184", dtype = "float32", shape = (12,))#candidate|1184|(12,)|var|float32
call_1181 = func_1180_call(var_1182,var_1183,var_1184,)
output = call_1181
func_1185 = relay.Function([var_1182,var_1183,var_1184,], output)
mutated_mod['func_1185'] = func_1185
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1194 = relay.var("var_1194", dtype = "uint32", shape = (3, 11))#candidate|1194|(3, 11)|var|uint32
var_1195 = relay.var("var_1195", dtype = "uint32", shape = (3, 11))#candidate|1195|(3, 11)|var|uint32
bop_1196 = relay.multiply(var_1194.astype('uint32'), relay.reshape(var_1195.astype('uint32'), relay.shape_of(var_1194))) # shape=(3, 11)
uop_1207 = relay.exp(var_1195.astype('float32')) # shape=(3, 11)
func_830_call = mod.get_global_var('func_830')
func_833_call = mutated_mod.get_global_var('func_833')
const_1213 = relay.const(-10, dtype = "int32")#candidate|1213|()|const|int32
var_1214 = relay.var("var_1214", dtype = "int32", shape = (12,))#candidate|1214|(12,)|var|int32
call_1212 = relay.TupleGetItem(func_830_call(relay.reshape(const_1213.astype('int32'), []), relay.reshape(var_1214.astype('int32'), [1, 12]), ), 0)
call_1215 = relay.TupleGetItem(func_833_call(relay.reshape(const_1213.astype('int32'), []), relay.reshape(var_1214.astype('int32'), [1, 12]), ), 0)
bop_1216 = relay.greater(uop_1207.astype('bool'), relay.reshape(bop_1196.astype('bool'), relay.shape_of(uop_1207))) # shape=(3, 11)
output = relay.Tuple([call_1212,const_1213,var_1214,bop_1216,])
output2 = relay.Tuple([call_1215,const_1213,var_1214,bop_1216,])
func_1220 = relay.Function([var_1194,var_1195,var_1214,], output)
mod['func_1220'] = func_1220
mod = relay.transform.InferType()(mod)
var_1221 = relay.var("var_1221", dtype = "uint32", shape = (3, 11))#candidate|1221|(3, 11)|var|uint32
var_1222 = relay.var("var_1222", dtype = "uint32", shape = (3, 11))#candidate|1222|(3, 11)|var|uint32
var_1223 = relay.var("var_1223", dtype = "int32", shape = (12,))#candidate|1223|(12,)|var|int32
output = func_1220(var_1221,var_1222,var_1223,)
func_1224 = relay.Function([var_1221,var_1222,var_1223,], output)
mutated_mod['func_1224'] = func_1224
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1226 = relay.const([[-3.719786,6.195492,1.652687,-5.876149,8.985797,-1.228818,1.887703,-6.546586,5.972804,4.743595,3.547540,-5.093283,6.042582,-0.353542,-8.375791,4.727413],[7.455650,0.872025,-6.382389,8.614098,-3.824266,-7.746336,8.909100,-5.181794,4.587896,-5.314904,-9.922209,2.475773,-1.619761,5.392778,-2.972445,-7.747915],[9.349821,-7.026425,9.981621,3.493404,4.417925,4.765145,-4.462639,7.747443,4.083221,2.913063,-4.995485,4.544387,-0.169743,-6.655845,0.562520,0.433340],[-3.828814,-2.243437,8.760173,5.154571,2.519877,6.144405,9.586387,4.802827,-8.977802,-8.428535,-6.522989,6.995737,1.540828,0.380162,-5.042526,6.330526],[-9.775610,-0.962160,-7.984196,4.563150,-7.111493,2.185236,4.990209,5.006567,5.469873,1.004539,6.785009,3.617343,-5.376578,-0.835583,2.919445,8.703011],[4.960212,0.834657,-6.848565,-0.578578,9.150778,-2.006635,-8.317499,-6.570252,2.841460,3.336461,-5.238067,5.638652,3.815047,-4.293588,-2.574660,9.819658],[0.967526,3.711667,9.917318,0.930238,6.906800,0.725043,8.256056,-4.841617,7.024166,6.943917,-8.699252,-9.274322,-2.880299,-1.000629,5.033827,9.525942],[-2.103381,-8.250720,-3.070974,-2.649379,9.315672,2.162193,1.185226,2.423580,-3.589469,-8.272118,-9.096006,-1.730203,4.231050,2.105814,-4.230199,-7.618104],[2.111933,-9.179918,-2.918240,8.108934,3.148695,-0.043157,-7.558999,1.541422,-3.160368,4.652422,7.638650,-3.216409,-1.958020,3.768137,-0.861238,1.503721],[-8.849031,4.333697,-3.084053,-3.179587,9.962468,-5.398577,-7.135380,-7.591803,-9.062891,5.351080,-7.673431,-0.310921,3.520127,7.368745,0.684128,1.448206],[0.899664,-1.126488,-2.147154,0.390116,9.412230,6.951265,7.369075,1.383254,-8.335913,3.878105,-3.790056,6.348215,7.175582,4.414592,-9.535242,-1.298916]], dtype = "float64")#candidate|1226|(11, 16)|const|float64
uop_1227 = relay.cos(const_1226.astype('float64')) # shape=(11, 16)
var_1230 = relay.var("var_1230", dtype = "float64", shape = (11, 16))#candidate|1230|(11, 16)|var|float64
bop_1231 = relay.logical_xor(uop_1227.astype('uint8'), relay.reshape(var_1230.astype('uint8'), relay.shape_of(uop_1227))) # shape=(11, 16)
output = relay.Tuple([bop_1231,])
output2 = relay.Tuple([bop_1231,])
func_1234 = relay.Function([var_1230,], output)
mod['func_1234'] = func_1234
mod = relay.transform.InferType()(mod)
var_1235 = relay.var("var_1235", dtype = "float64", shape = (11, 16))#candidate|1235|(11, 16)|var|float64
output = func_1234(var_1235)
func_1236 = relay.Function([var_1235], output)
mutated_mod['func_1236'] = func_1236
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1241 = relay.var("var_1241", dtype = "int16", shape = (9, 6, 8))#candidate|1241|(9, 6, 8)|var|int16
var_1242 = relay.var("var_1242", dtype = "int16", shape = (9, 6, 8))#candidate|1242|(9, 6, 8)|var|int16
bop_1243 = relay.greater(var_1241.astype('bool'), relay.reshape(var_1242.astype('bool'), relay.shape_of(var_1241))) # shape=(9, 6, 8)
uop_1248 = relay.erf(var_1241.astype('float32')) # shape=(9, 6, 8)
bop_1250 = relay.right_shift(uop_1248.astype('uint8'), relay.reshape(bop_1243.astype('uint8'), relay.shape_of(uop_1248))) # shape=(9, 6, 8)
output = bop_1250
output2 = bop_1250
func_1254 = relay.Function([var_1241,var_1242,], output)
mod['func_1254'] = func_1254
mod = relay.transform.InferType()(mod)
var_1255 = relay.var("var_1255", dtype = "int16", shape = (9, 6, 8))#candidate|1255|(9, 6, 8)|var|int16
var_1256 = relay.var("var_1256", dtype = "int16", shape = (9, 6, 8))#candidate|1256|(9, 6, 8)|var|int16
output = func_1254(var_1255,var_1256,)
func_1257 = relay.Function([var_1255,var_1256,], output)
mutated_mod['func_1257'] = func_1257
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1262 = relay.const([[-1.433658,-5.189219,7.784873],[-6.261098,-1.049144,2.719369],[8.443771,-7.799015,6.708189],[0.140101,2.496340,8.979555],[-0.710025,-4.246860,-4.250213],[-7.660562,4.374382,6.726541],[-9.610676,4.560796,-8.252934],[8.447249,-3.017172,4.090965],[4.960861,9.622715,3.636116],[-8.634721,-0.379748,-0.294341],[-6.030749,5.929038,8.070126],[-0.955902,-0.164172,-3.805127],[6.565613,7.015413,5.231004]], dtype = "float64")#candidate|1262|(13, 3)|const|float64
uop_1263 = relay.sin(const_1262.astype('float64')) # shape=(13, 3)
uop_1265 = relay.sinh(const_1262.astype('float64')) # shape=(13, 3)
bop_1267 = relay.not_equal(uop_1263.astype('bool'), relay.reshape(uop_1265.astype('bool'), relay.shape_of(uop_1263))) # shape=(13, 3)
output = relay.Tuple([bop_1267,])
output2 = relay.Tuple([bop_1267,])
func_1270 = relay.Function([], output)
mod['func_1270'] = func_1270
mod = relay.transform.InferType()(mod)
output = func_1270()
func_1271 = relay.Function([], output)
mutated_mod['func_1271'] = func_1271
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1306 = relay.var("var_1306", dtype = "uint16", shape = (7, 15))#candidate|1306|(7, 15)|var|uint16
var_1307 = relay.var("var_1307", dtype = "uint16", shape = (7, 15))#candidate|1307|(7, 15)|var|uint16
bop_1308 = relay.logical_xor(var_1306.astype('uint16'), relay.reshape(var_1307.astype('uint16'), relay.shape_of(var_1306))) # shape=(7, 15)
bop_1317 = relay.bitwise_xor(bop_1308.astype('uint64'), relay.reshape(var_1307.astype('uint64'), relay.shape_of(bop_1308))) # shape=(7, 15)
bop_1321 = relay.right_shift(var_1307.astype('uint64'), relay.reshape(var_1306.astype('uint64'), relay.shape_of(var_1307))) # shape=(7, 15)
var_1324 = relay.var("var_1324", dtype = "uint64", shape = (7, 15))#candidate|1324|(7, 15)|var|uint64
bop_1325 = relay.not_equal(bop_1321.astype('bool'), relay.reshape(var_1324.astype('bool'), relay.shape_of(bop_1321))) # shape=(7, 15)
func_1220_call = mod.get_global_var('func_1220')
func_1224_call = mutated_mod.get_global_var('func_1224')
var_1330 = relay.var("var_1330", dtype = "uint32", shape = (33,))#candidate|1330|(33,)|var|uint32
const_1331 = relay.const([5,-4,3,-4,8,-5,5,7,6,-2,-8,-7], dtype = "int32")#candidate|1331|(12,)|const|int32
call_1329 = relay.TupleGetItem(func_1220_call(relay.reshape(var_1330.astype('uint32'), [3, 11]), relay.reshape(var_1330.astype('uint32'), [3, 11]), relay.reshape(const_1331.astype('int32'), [12,]), ), 1)
call_1332 = relay.TupleGetItem(func_1224_call(relay.reshape(var_1330.astype('uint32'), [3, 11]), relay.reshape(var_1330.astype('uint32'), [3, 11]), relay.reshape(const_1331.astype('int32'), [12,]), ), 1)
func_1054_call = mod.get_global_var('func_1054')
func_1057_call = mutated_mod.get_global_var('func_1057')
var_1338 = relay.var("var_1338", dtype = "float32", shape = (208,))#candidate|1338|(208,)|var|float32
call_1337 = relay.TupleGetItem(func_1054_call(relay.reshape(var_1338.astype('float32'), [13, 16]), relay.reshape(const_1331.astype('float32'), [12,]), ), 2)
call_1339 = relay.TupleGetItem(func_1057_call(relay.reshape(var_1338.astype('float32'), [13, 16]), relay.reshape(const_1331.astype('float32'), [12,]), ), 2)
output = relay.Tuple([bop_1317,bop_1325,call_1329,var_1330,const_1331,call_1337,var_1338,])
output2 = relay.Tuple([bop_1317,bop_1325,call_1332,var_1330,const_1331,call_1339,var_1338,])
func_1341 = relay.Function([var_1306,var_1307,var_1324,var_1330,var_1338,], output)
mod['func_1341'] = func_1341
mod = relay.transform.InferType()(mod)
var_1342 = relay.var("var_1342", dtype = "uint16", shape = (7, 15))#candidate|1342|(7, 15)|var|uint16
var_1343 = relay.var("var_1343", dtype = "uint16", shape = (7, 15))#candidate|1343|(7, 15)|var|uint16
var_1344 = relay.var("var_1344", dtype = "uint64", shape = (7, 15))#candidate|1344|(7, 15)|var|uint64
var_1345 = relay.var("var_1345", dtype = "uint32", shape = (33,))#candidate|1345|(33,)|var|uint32
var_1346 = relay.var("var_1346", dtype = "float32", shape = (208,))#candidate|1346|(208,)|var|float32
output = func_1341(var_1342,var_1343,var_1344,var_1345,var_1346,)
func_1347 = relay.Function([var_1342,var_1343,var_1344,var_1345,var_1346,], output)
mutated_mod['func_1347'] = func_1347
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1270_call = mod.get_global_var('func_1270')
func_1271_call = mutated_mod.get_global_var('func_1271')
call_1395 = relay.TupleGetItem(func_1270_call(), 0)
call_1396 = relay.TupleGetItem(func_1271_call(), 0)
var_1404 = relay.var("var_1404", dtype = "bool", shape = (13, 3))#candidate|1404|(13, 3)|var|bool
bop_1405 = relay.right_shift(call_1395.astype('uint8'), relay.reshape(var_1404.astype('uint8'), relay.shape_of(call_1395))) # shape=(13, 3)
bop_1408 = relay.right_shift(call_1396.astype('uint8'), relay.reshape(var_1404.astype('uint8'), relay.shape_of(call_1396))) # shape=(13, 3)
output = bop_1405
output2 = bop_1408
func_1422 = relay.Function([var_1404,], output)
mod['func_1422'] = func_1422
mod = relay.transform.InferType()(mod)
var_1423 = relay.var("var_1423", dtype = "bool", shape = (13, 3))#candidate|1423|(13, 3)|var|bool
output = func_1422(var_1423)
func_1424 = relay.Function([var_1423], output)
mutated_mod['func_1424'] = func_1424
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1270_call = mod.get_global_var('func_1270')
func_1271_call = mutated_mod.get_global_var('func_1271')
call_1434 = relay.TupleGetItem(func_1270_call(), 0)
call_1435 = relay.TupleGetItem(func_1271_call(), 0)
func_818_call = mod.get_global_var('func_818')
func_822_call = mutated_mod.get_global_var('func_822')
var_1440 = relay.var("var_1440", dtype = "int8", shape = (112,))#candidate|1440|(112,)|var|int8
var_1441 = relay.var("var_1441", dtype = "int8", shape = (672,))#candidate|1441|(672,)|var|int8
call_1439 = relay.TupleGetItem(func_818_call(relay.reshape(var_1440.astype('int8'), [7, 1, 16]), relay.reshape(var_1441.astype('int8'), [7, 6, 16]), ), 0)
call_1442 = relay.TupleGetItem(func_822_call(relay.reshape(var_1440.astype('int8'), [7, 1, 16]), relay.reshape(var_1441.astype('int8'), [7, 6, 16]), ), 0)
uop_1463 = relay.atan(var_1440.astype('float32')) # shape=(112,)
var_1468 = relay.var("var_1468", dtype = "int8", shape = (112,))#candidate|1468|(112,)|var|int8
bop_1469 = relay.equal(var_1440.astype('bool'), relay.reshape(var_1468.astype('bool'), relay.shape_of(var_1440))) # shape=(112,)
const_1481 = relay.const([8.069067,0.230365,3.862978,2.729117,6.302060,-7.151699,1.154079,-8.157709,-1.193097,1.529309,-6.513012,-5.010657,6.449215,-2.630696,1.780684,-3.204969,2.561187,-9.473934,-3.436306,-8.476826,-4.156937,-3.896311,-6.635640,-3.949814,-2.162797,0.105703,5.061915,0.703032,8.108270,5.536294,-0.848926,7.243437,7.314739,-7.751416,-3.462163,-3.184219,-4.407529,-7.014017,7.277260,8.158349,-5.149065,-1.309658,-8.695605,-5.561324,6.748981,2.351260,8.695204,3.055775,-2.233138,3.357056,-6.265672,-1.675121,-2.715073,2.222317,5.896212,6.528616,-4.496149,0.717131,-8.842226,-1.683224,-4.074815,8.731867,5.395930,-2.176926,-1.209063,-0.536295,6.698390,9.531204,6.100120,-5.935678,-3.447922,-9.277204,8.497821,9.213819,7.168323,-5.026586,9.785140,6.891121,-7.784595,6.291794,0.163123,-5.382607,-9.364456,-6.243866,-1.336381,-4.241808,2.294589,8.111411,7.212602,-0.553140,1.704729,-2.478360,-6.297653,5.678429,5.518762,9.683504,-8.505052,-3.063780,-2.810726,-6.418521,2.295328,4.406048,5.698145,2.134211,7.717213,0.279340,-4.793380,-8.139832,6.999170,-3.302007,-3.321161,-5.517371], dtype = "float32")#candidate|1481|(112,)|const|float32
bop_1482 = relay.bitwise_or(uop_1463.astype('int32'), relay.reshape(const_1481.astype('int32'), relay.shape_of(uop_1463))) # shape=(112,)
bop_1487 = relay.subtract(uop_1463.astype('uint32'), relay.reshape(bop_1469.astype('uint32'), relay.shape_of(uop_1463))) # shape=(112,)
uop_1490 = relay.rsqrt(const_1481.astype('float64')) # shape=(112,)
output = relay.Tuple([call_1434,call_1439,var_1441,bop_1482,bop_1487,uop_1490,])
output2 = relay.Tuple([call_1435,call_1442,var_1441,bop_1482,bop_1487,uop_1490,])
func_1494 = relay.Function([var_1440,var_1441,var_1468,], output)
mod['func_1494'] = func_1494
mod = relay.transform.InferType()(mod)
mutated_mod['func_1494'] = func_1494
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1494_call = mutated_mod.get_global_var('func_1494')
var_1496 = relay.var("var_1496", dtype = "int8", shape = (112,))#candidate|1496|(112,)|var|int8
var_1497 = relay.var("var_1497", dtype = "int8", shape = (672,))#candidate|1497|(672,)|var|int8
var_1498 = relay.var("var_1498", dtype = "int8", shape = (112,))#candidate|1498|(112,)|var|int8
call_1495 = func_1494_call(var_1496,var_1497,var_1498,)
output = call_1495
func_1499 = relay.Function([var_1496,var_1497,var_1498,], output)
mutated_mod['func_1499'] = func_1499
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1270_call = mod.get_global_var('func_1270')
func_1271_call = mutated_mod.get_global_var('func_1271')
call_1600 = relay.TupleGetItem(func_1270_call(), 0)
call_1601 = relay.TupleGetItem(func_1271_call(), 0)
output = call_1600
output2 = call_1601
func_1607 = relay.Function([], output)
mod['func_1607'] = func_1607
mod = relay.transform.InferType()(mod)
mutated_mod['func_1607'] = func_1607
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1607_call = mutated_mod.get_global_var('func_1607')
call_1608 = func_1607_call()
output = call_1608
func_1609 = relay.Function([], output)
mutated_mod['func_1609'] = func_1609
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1626 = relay.var("var_1626", dtype = "float64", shape = (14, 5))#candidate|1626|(14, 5)|var|float64
uop_1627 = relay.asin(var_1626.astype('float64')) # shape=(14, 5)
output = relay.Tuple([uop_1627,])
output2 = relay.Tuple([uop_1627,])
func_1629 = relay.Function([var_1626,], output)
mod['func_1629'] = func_1629
mod = relay.transform.InferType()(mod)
var_1630 = relay.var("var_1630", dtype = "float64", shape = (14, 5))#candidate|1630|(14, 5)|var|float64
output = func_1629(var_1630)
func_1631 = relay.Function([var_1630], output)
mutated_mod['func_1631'] = func_1631
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1645 = relay.var("var_1645", dtype = "float64", shape = (6, 7, 16))#candidate|1645|(6, 7, 16)|var|float64
var_1646 = relay.var("var_1646", dtype = "float64", shape = (6, 7, 16))#candidate|1646|(6, 7, 16)|var|float64
bop_1647 = relay.mod(var_1645.astype('float64'), relay.reshape(var_1646.astype('float64'), relay.shape_of(var_1645))) # shape=(6, 7, 16)
func_818_call = mod.get_global_var('func_818')
func_822_call = mutated_mod.get_global_var('func_822')
const_1652 = relay.const([4,7,-3,-8,10,6,-6,4,1,-1,-1,1,9,-4,-4,8,-4,6,9,-6,7,-6,-10,-5,8,7,-4,9,8,4,-8,-10,9,-10,1,-9,7,-1,8,2,-9,5,-9,7,-8,9,-5,-8,6,3,3,6,-9,-1,-1,5,1,-3,4,10,-1,8,8,10,-4,-1,-5,9,4,-3,2,-6,9,3,-1,7,-2,10,3,-5,-3,7,5,-1,5,-7,3,-10,-7,8,-8,2,2,-3,3,3,6,1,-6,-2,8,8,5,4,4,9,-8,3,6,4,-2,3], dtype = "int8")#candidate|1652|(112,)|const|int8
call_1651 = relay.TupleGetItem(func_818_call(relay.reshape(const_1652.astype('int8'), [7, 1, 16]), relay.reshape(var_1646.astype('int8'), [7, 6, 16]), ), 0)
call_1653 = relay.TupleGetItem(func_822_call(relay.reshape(const_1652.astype('int8'), [7, 1, 16]), relay.reshape(var_1646.astype('int8'), [7, 6, 16]), ), 0)
bop_1655 = relay.minimum(bop_1647.astype('float64'), relay.reshape(call_1651.astype('float64'), relay.shape_of(bop_1647))) # shape=(6, 7, 16)
bop_1658 = relay.minimum(bop_1647.astype('float64'), relay.reshape(call_1653.astype('float64'), relay.shape_of(bop_1647))) # shape=(6, 7, 16)
uop_1659 = relay.sqrt(var_1646.astype('float64')) # shape=(6, 7, 16)
bop_1661 = relay.bitwise_or(var_1645.astype('uint16'), relay.reshape(uop_1659.astype('uint16'), relay.shape_of(var_1645))) # shape=(6, 7, 16)
output = relay.Tuple([const_1652,bop_1655,bop_1661,])
output2 = relay.Tuple([const_1652,bop_1658,bop_1661,])
func_1668 = relay.Function([var_1645,var_1646,], output)
mod['func_1668'] = func_1668
mod = relay.transform.InferType()(mod)
mutated_mod['func_1668'] = func_1668
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1668_call = mutated_mod.get_global_var('func_1668')
var_1670 = relay.var("var_1670", dtype = "float64", shape = (6, 7, 16))#candidate|1670|(6, 7, 16)|var|float64
var_1671 = relay.var("var_1671", dtype = "float64", shape = (6, 7, 16))#candidate|1671|(6, 7, 16)|var|float64
call_1669 = func_1668_call(var_1670,var_1671,)
output = call_1669
func_1672 = relay.Function([var_1670,var_1671,], output)
mutated_mod['func_1672'] = func_1672
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1270_call = mod.get_global_var('func_1270')
func_1271_call = mutated_mod.get_global_var('func_1271')
call_1681 = relay.TupleGetItem(func_1270_call(), 0)
call_1682 = relay.TupleGetItem(func_1271_call(), 0)
output = call_1681
output2 = call_1682
func_1684 = relay.Function([], output)
mod['func_1684'] = func_1684
mod = relay.transform.InferType()(mod)
mutated_mod['func_1684'] = func_1684
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1684_call = mutated_mod.get_global_var('func_1684')
call_1685 = func_1684_call()
output = call_1685
func_1686 = relay.Function([], output)
mutated_mod['func_1686'] = func_1686
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1751 = relay.var("var_1751", dtype = "int64", shape = (8, 1, 15))#candidate|1751|(8, 1, 15)|var|int64
var_1752 = relay.var("var_1752", dtype = "int64", shape = (8, 1, 15))#candidate|1752|(8, 1, 15)|var|int64
bop_1753 = relay.less(var_1751.astype('bool'), relay.reshape(var_1752.astype('bool'), relay.shape_of(var_1751))) # shape=(8, 1, 15)
output = relay.Tuple([bop_1753,])
output2 = relay.Tuple([bop_1753,])
func_1765 = relay.Function([var_1751,var_1752,], output)
mod['func_1765'] = func_1765
mod = relay.transform.InferType()(mod)
mutated_mod['func_1765'] = func_1765
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1765_call = mutated_mod.get_global_var('func_1765')
var_1767 = relay.var("var_1767", dtype = "int64", shape = (8, 1, 15))#candidate|1767|(8, 1, 15)|var|int64
var_1768 = relay.var("var_1768", dtype = "int64", shape = (8, 1, 15))#candidate|1768|(8, 1, 15)|var|int64
call_1766 = func_1765_call(var_1767,var_1768,)
output = call_1766
func_1769 = relay.Function([var_1767,var_1768,], output)
mutated_mod['func_1769'] = func_1769
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1607_call = mod.get_global_var('func_1607')
func_1609_call = mutated_mod.get_global_var('func_1609')
call_1845 = func_1607_call()
call_1846 = func_1607_call()
output = call_1845
output2 = call_1846
func_1854 = relay.Function([], output)
mod['func_1854'] = func_1854
mod = relay.transform.InferType()(mod)
mutated_mod['func_1854'] = func_1854
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1854_call = mutated_mod.get_global_var('func_1854')
call_1855 = func_1854_call()
output = call_1855
func_1856 = relay.Function([], output)
mutated_mod['func_1856'] = func_1856
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1607_call = mod.get_global_var('func_1607')
func_1609_call = mutated_mod.get_global_var('func_1609')
call_1881 = func_1607_call()
call_1882 = func_1607_call()
output = relay.Tuple([call_1881,])
output2 = relay.Tuple([call_1882,])
func_1883 = relay.Function([], output)
mod['func_1883'] = func_1883
mod = relay.transform.InferType()(mod)
output = func_1883()
func_1884 = relay.Function([], output)
mutated_mod['func_1884'] = func_1884
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1898 = relay.var("var_1898", dtype = "int16", shape = ())#candidate|1898|()|var|int16
var_1899 = relay.var("var_1899", dtype = "int16", shape = (10, 4, 15))#candidate|1899|(10, 4, 15)|var|int16
bop_1900 = relay.bitwise_and(var_1898.astype('int16'), var_1899.astype('int16')) # shape=(10, 4, 15)
output = bop_1900
output2 = bop_1900
func_1905 = relay.Function([var_1898,var_1899,], output)
mod['func_1905'] = func_1905
mod = relay.transform.InferType()(mod)
mutated_mod['func_1905'] = func_1905
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1905_call = mutated_mod.get_global_var('func_1905')
var_1907 = relay.var("var_1907", dtype = "int16", shape = ())#candidate|1907|()|var|int16
var_1908 = relay.var("var_1908", dtype = "int16", shape = (10, 4, 15))#candidate|1908|(10, 4, 15)|var|int16
call_1906 = func_1905_call(var_1907,var_1908,)
output = call_1906
func_1909 = relay.Function([var_1907,var_1908,], output)
mutated_mod['func_1909'] = func_1909
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1916 = relay.var("var_1916", dtype = "float64", shape = (14, 4))#candidate|1916|(14, 4)|var|float64
uop_1917 = relay.atan(var_1916.astype('float64')) # shape=(14, 4)
bop_1923 = relay.floor_mod(var_1916.astype('float32'), relay.reshape(uop_1917.astype('float32'), relay.shape_of(var_1916))) # shape=(14, 4)
bop_1928 = relay.bitwise_xor(uop_1917.astype('int64'), relay.reshape(bop_1923.astype('int64'), relay.shape_of(uop_1917))) # shape=(14, 4)
bop_1934 = relay.not_equal(var_1916.astype('bool'), relay.reshape(bop_1923.astype('bool'), relay.shape_of(var_1916))) # shape=(14, 4)
var_1948 = relay.var("var_1948", dtype = "bool", shape = (14, 4))#candidate|1948|(14, 4)|var|bool
bop_1949 = relay.power(bop_1934.astype('float32'), relay.reshape(var_1948.astype('float32'), relay.shape_of(bop_1934))) # shape=(14, 4)
func_855_call = mod.get_global_var('func_855')
func_857_call = mutated_mod.get_global_var('func_857')
var_1953 = relay.var("var_1953", dtype = "float64", shape = (792,))#candidate|1953|(792,)|var|float64
call_1952 = func_855_call(relay.reshape(var_1953.astype('float64'), [9, 8, 11]))
call_1954 = func_855_call(relay.reshape(var_1953.astype('float64'), [9, 8, 11]))
var_1955 = relay.var("var_1955", dtype = "bool", shape = (14, 4))#candidate|1955|(14, 4)|var|bool
bop_1956 = relay.left_shift(bop_1934.astype('int16'), relay.reshape(var_1955.astype('int16'), relay.shape_of(bop_1934))) # shape=(14, 4)
bop_1961 = relay.maximum(var_1955.astype('int8'), relay.reshape(var_1916.astype('int8'), relay.shape_of(var_1955))) # shape=(14, 4)
uop_1964 = relay.asin(bop_1956.astype('float64')) # shape=(14, 4)
output = relay.Tuple([bop_1928,bop_1949,call_1952,var_1953,bop_1961,uop_1964,])
output2 = relay.Tuple([bop_1928,bop_1949,call_1954,var_1953,bop_1961,uop_1964,])
func_1970 = relay.Function([var_1916,var_1948,var_1953,var_1955,], output)
mod['func_1970'] = func_1970
mod = relay.transform.InferType()(mod)
var_1971 = relay.var("var_1971", dtype = "float64", shape = (14, 4))#candidate|1971|(14, 4)|var|float64
var_1972 = relay.var("var_1972", dtype = "bool", shape = (14, 4))#candidate|1972|(14, 4)|var|bool
var_1973 = relay.var("var_1973", dtype = "float64", shape = (792,))#candidate|1973|(792,)|var|float64
var_1974 = relay.var("var_1974", dtype = "bool", shape = (14, 4))#candidate|1974|(14, 4)|var|bool
output = func_1970(var_1971,var_1972,var_1973,var_1974,)
func_1975 = relay.Function([var_1971,var_1972,var_1973,var_1974,], output)
mutated_mod['func_1975'] = func_1975
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2004 = relay.const([[-5.877318,-4.359052,-3.363642,4.898708,5.338992,-4.794181,-7.221069,-8.783154,-7.451451,0.824113,7.576948,9.378787,-0.268589,9.934941,-5.594061,7.053157],[-0.497179,3.015461,-0.176566,-4.594245,8.063772,-5.400273,4.499749,8.487591,-6.173883,-1.732931,-0.369358,-3.859716,5.939618,5.093797,7.772474,-4.831963],[-2.981378,-7.513782,-6.956358,-8.381735,1.059781,0.002241,0.630458,-2.390530,4.372403,-0.846243,9.105387,-0.538602,-4.392268,-0.874036,4.789993,1.622713],[-0.930775,-9.646594,3.222747,-2.582678,9.198067,2.746821,-7.675471,5.135036,-1.495276,7.036415,-7.186849,-8.948356,-3.348417,-7.998264,2.749226,3.962119],[-2.697177,1.785365,1.350139,-7.631643,5.120785,5.154338,-3.781625,-1.722099,-5.670536,-5.062836,8.783610,5.094612,0.177494,5.771296,-4.766172,-9.536557],[-9.911567,-4.933488,-9.086392,5.802339,-1.230687,6.462627,5.930232,-4.065495,-1.126144,-0.557668,5.763237,1.340547,-1.649598,-6.294408,0.027533,5.075758],[-0.521712,0.955733,-0.904622,-1.841601,-2.933889,4.494536,4.014069,3.234865,-2.055293,-3.777173,-8.457349,-9.360604,-4.331943,7.951377,-1.870747,-4.660453],[-4.883309,0.409118,-5.710215,2.197167,6.093748,-5.584329,-1.515259,-7.687340,4.949782,4.040990,-4.387645,-0.587963,-1.877815,8.421360,-3.441020,5.828450],[5.160260,-2.783887,-4.543191,-6.151003,-9.619835,1.518902,7.173844,-5.478339,-1.287470,-8.839170,1.881732,6.991220,-6.706353,-7.311258,-5.250021,4.300345],[-5.297948,3.737998,7.878018,8.665608,-4.839260,3.343327,-3.263485,4.999012,5.303290,-6.246978,-6.759317,-0.534939,-5.406959,-1.759208,-7.434766,-8.455951],[6.141818,8.752203,-5.425795,-6.236288,2.869210,-2.059907,7.391993,-2.079086,8.631572,-7.348682,6.814961,-8.167001,4.058416,-4.495926,-7.336370,3.123264],[3.850922,-1.140482,1.843274,-5.276084,-3.116783,-7.460177,-5.668714,-2.494590,7.277696,-5.998124,0.428127,-5.223173,3.918136,-8.555829,-2.784690,5.836946]], dtype = "float32")#candidate|2004|(12, 16)|const|float32
uop_2005 = relay.log2(const_2004.astype('float32')) # shape=(12, 16)
output = relay.Tuple([uop_2005,])
output2 = relay.Tuple([uop_2005,])
func_2012 = relay.Function([], output)
mod['func_2012'] = func_2012
mod = relay.transform.InferType()(mod)
output = func_2012()
func_2013 = relay.Function([], output)
mutated_mod['func_2013'] = func_2013
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1854_call = mod.get_global_var('func_1854')
func_1856_call = mutated_mod.get_global_var('func_1856')
call_2059 = func_1854_call()
call_2060 = func_1854_call()
func_1494_call = mod.get_global_var('func_1494')
func_1499_call = mutated_mod.get_global_var('func_1499')
var_2068 = relay.var("var_2068", dtype = "int8", shape = (112,))#candidate|2068|(112,)|var|int8
const_2069 = relay.const([5,-7,-2,-10,4,-7,-1,-6,3,-8,-4,-6,10,4,-2,-8,-2,-1,5,5,1,10,3,7,-1,-1,-9,-4,4,9,-4,-5,2,-5,7,-10,-3,10,-10,10,-1,-8,9,-1,3,10,-7,-6,-1,9,2,-6,-2,9,-1,-5,1,5,7,-4,6,-2,9,1,10,9,9,-4,-2,-3,3,7,1,-5,-8,3,-3,6,-2,3,8,3,-7,-8,-2,-2,-1,9,3,-2,-5,6,9,-1,10,2,-3,-5,2,9,-6,-6,-4,-1,-3,7,-10,-7,6,10,9,1,-5,-4,2,8,8,7,3,8,-2,5,9,-1,-7,10,2,-9,10,3,-1,-9,9,1,-5,9,6,-7,5,10,10,-7,1,-9,-1,10,5,1,7,-7,2,10,-9,1,-7,-5,10,7,-10,-2,-1,5,4,-2,-9,-5,-7,9,4,-2,-2,7,-1,-3,3,3,8,1,-5,4,4,-5,6,8,5,-10,9,6,2,-2,-3,-8,-3,2,2,4,-7,8,2,-3,6,-3,-9,-4,8,-5,-10,-3,-3,-3,-3,2,5,-5,-6,5,2,-8,6,-3,9,-7,-4,-4,-7,4,3,2,-8,8,-9,-9,-10,3,8,-2,5,-7,-10,-8,7,7,-9,-1,-9,-4,-3,-9,-9,-1,-10,5,-2,8,3,-1,7,-2,8,-7,2,5,-4,8,-1,-2,4,8,-10,-2,-2,3,10,8,-1,-6,6,-10,-8,-6,9,-9,5,-1,-4,8,-5,-5,-8,-2,10,4,9,-8,4,5,-10,-8,-2,10,3,3,-10,-2,5,-7,-2,-1,-5,-8,-1,-2,-7,-5,-1,-8,2,-2,-4,-1,8,2,-1,1,-9,4,-1,-10,9,1,2,-2,-3,-9,9,-5,-9,7,9,-5,6,-8,-6,-9,-6,4,-2,-2,-3,3,10,8,-5,7,-4,4,-2,9,3,-4,10,6,-4,-8,-9,8,-9,9,3,1,8,7,10,-9,-6,3,-1,-10,7,-2,9,6,-6,5,10,-5,3,-10,10,5,4,-3,-7,2,3,10,3,3,2,-9,5,-3,-7,-1,-4,-6,4,-5,-4,-7,-8,10,-5,-3,10,9,7,4,-7,-3,-3,1,-8,-10,-3,9,4,-5,10,-5,-7,8,8,-7,8,8,-9,-9,-7,-1,-8,2,-10,10,-4,-2,1,6,-1,8,5,1,7,2,10,4,4,9,-4,3,6,-8,5,-10,-7,10,-4,-5,-10,5,-10,8,-9,-7,-6,9,-6,-2,-1,1,-1,-9,1,-6,-2,-3,-1,8,7,-3,8,6,-10,-8,1,8,-10,6,2,-2,-2,1,-5,-3,6,7,7,7,4,-8,-5,-8,5,6,-4,6,-5,-9,-5,3,10,3,-5,-7,-6,-5,-3,-5,7,-10,2,-8,10,-7,2,6,10,-3,2,6,-7,10,-6,-2,10,3,-7,-7,10,4,4,6,-1,-2,1,4,9,-9,1,-7,5,-3,2,-9,-5,-4,3,4,-1,9,-4,-1,9,9,3,-10,-7,9,4,10,1,10,-6,-1,1,-8,-5,10,-10,3,-5,10,-8,1,-8,-1,7,-6,6,-1,-9,1,-9,-5,10,-10,1,-8,10,-10,2,-9,3,-3,3,7,2,-6,-1,-10,9,9,-4,10,6,-4,3,5,-1,6,-2,-7,-1,7,-4,-4,-9,-4,7,6,-3,-2,-6,-2,-7,2,-9,6,-2,1,-2,-8,7,-3,-5,10,3,-4,-8,5,-5,5,-8,6,-3,-5,-4,-5,6,-7,-9,10], dtype = "int8")#candidate|2069|(672,)|const|int8
call_2067 = relay.TupleGetItem(func_1494_call(relay.reshape(var_2068.astype('int8'), [112,]), relay.reshape(const_2069.astype('int8'), [672,]), relay.reshape(var_2068.astype('int8'), [112,]), ), 3)
call_2070 = relay.TupleGetItem(func_1499_call(relay.reshape(var_2068.astype('int8'), [112,]), relay.reshape(const_2069.astype('int8'), [672,]), relay.reshape(var_2068.astype('int8'), [112,]), ), 3)
func_1220_call = mod.get_global_var('func_1220')
func_1224_call = mutated_mod.get_global_var('func_1224')
var_2089 = relay.var("var_2089", dtype = "uint32", shape = (33,))#candidate|2089|(33,)|var|uint32
var_2090 = relay.var("var_2090", dtype = "int32", shape = (1, 12))#candidate|2090|(1, 12)|var|int32
call_2088 = relay.TupleGetItem(func_1220_call(relay.reshape(var_2089.astype('uint32'), [3, 11]), relay.reshape(var_2089.astype('uint32'), [3, 11]), relay.reshape(var_2090.astype('int32'), [12,]), ), 2)
call_2091 = relay.TupleGetItem(func_1224_call(relay.reshape(var_2089.astype('uint32'), [3, 11]), relay.reshape(var_2089.astype('uint32'), [3, 11]), relay.reshape(var_2090.astype('int32'), [12,]), ), 2)
func_443_call = mod.get_global_var('func_443')
func_445_call = mutated_mod.get_global_var('func_445')
call_2102 = relay.TupleGetItem(func_443_call(relay.reshape(call_2088.astype('float32'), [1, 12])), 1)
call_2103 = relay.TupleGetItem(func_445_call(relay.reshape(call_2088.astype('float32'), [1, 12])), 1)
output = relay.Tuple([call_2059,call_2067,var_2068,const_2069,call_2088,var_2089,var_2090,call_2102,])
output2 = relay.Tuple([call_2060,call_2070,var_2068,const_2069,call_2091,var_2089,var_2090,call_2103,])
func_2104 = relay.Function([var_2068,var_2089,var_2090,], output)
mod['func_2104'] = func_2104
mod = relay.transform.InferType()(mod)
mutated_mod['func_2104'] = func_2104
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2104_call = mutated_mod.get_global_var('func_2104')
var_2106 = relay.var("var_2106", dtype = "int8", shape = (112,))#candidate|2106|(112,)|var|int8
var_2107 = relay.var("var_2107", dtype = "uint32", shape = (33,))#candidate|2107|(33,)|var|uint32
var_2108 = relay.var("var_2108", dtype = "int32", shape = (1, 12))#candidate|2108|(1, 12)|var|int32
call_2105 = func_2104_call(var_2106,var_2107,var_2108,)
output = call_2105
func_2109 = relay.Function([var_2106,var_2107,var_2108,], output)
mutated_mod['func_2109'] = func_2109
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1854_call = mod.get_global_var('func_1854')
func_1856_call = mutated_mod.get_global_var('func_1856')
call_2148 = func_1854_call()
call_2149 = func_1854_call()
output = call_2148
output2 = call_2149
func_2169 = relay.Function([], output)
mod['func_2169'] = func_2169
mod = relay.transform.InferType()(mod)
output = func_2169()
func_2170 = relay.Function([], output)
mutated_mod['func_2170'] = func_2170
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2169_call = mod.get_global_var('func_2169')
func_2170_call = mutated_mod.get_global_var('func_2170')
call_2183 = func_2169_call()
call_2184 = func_2169_call()
func_1607_call = mod.get_global_var('func_1607')
func_1609_call = mutated_mod.get_global_var('func_1609')
call_2185 = func_1607_call()
call_2186 = func_1607_call()
func_1629_call = mod.get_global_var('func_1629')
func_1631_call = mutated_mod.get_global_var('func_1631')
const_2188 = relay.const([-0.790848,-3.369018,0.371847,-3.748142,-4.401666,6.201619,7.432641,-9.862895,3.661001,-8.345107,-5.431836,4.332408,-9.535362,3.170749,8.214714,-8.243593,3.952440,0.084111,-5.043206,2.128436,-3.774364,-9.274007,0.279953,-3.748170,1.106251,0.988673,-2.441260,6.679872,2.192448,5.259575,8.391243,-7.012104,-7.234020,2.446747,4.622950,6.198866,-4.483499,8.861608,3.813521,-5.822355,-7.167955,-2.814591,-4.377985,-5.737280,-9.233083,-6.227209,1.304659,-5.426088,-3.153008,-4.728691,2.586353,6.831288,9.662682,0.214740,-6.557033,8.412558,4.157043,-4.337933,0.441232,1.584680,-8.660150,-7.045498,-4.859515,4.075190,-6.373740,-2.783125,-3.246261,3.339705,-7.564813,-1.908779], dtype = "float64")#candidate|2188|(70,)|const|float64
call_2187 = relay.TupleGetItem(func_1629_call(relay.reshape(const_2188.astype('float64'), [14, 5])), 0)
call_2189 = relay.TupleGetItem(func_1631_call(relay.reshape(const_2188.astype('float64'), [14, 5])), 0)
bop_2197 = relay.add(call_2183.astype('float32'), relay.reshape(call_2185.astype('float32'), relay.shape_of(call_2183))) # shape=(13, 3)
bop_2200 = relay.add(call_2184.astype('float32'), relay.reshape(call_2186.astype('float32'), relay.shape_of(call_2184))) # shape=(13, 3)
bop_2209 = relay.left_shift(call_2185.astype('int16'), relay.reshape(call_2183.astype('int16'), relay.shape_of(call_2185))) # shape=(13, 3)
bop_2212 = relay.left_shift(call_2186.astype('int16'), relay.reshape(call_2184.astype('int16'), relay.shape_of(call_2186))) # shape=(13, 3)
func_443_call = mod.get_global_var('func_443')
func_445_call = mutated_mod.get_global_var('func_445')
const_2220 = relay.const([-8.723512,3.955301,-0.915981,-7.817833,7.710288,1.418653,1.302069,-1.684420,-3.345485,-5.529158,-6.197240,-3.197817], dtype = "float32")#candidate|2220|(12,)|const|float32
call_2219 = relay.TupleGetItem(func_443_call(relay.reshape(const_2220.astype('float32'), [1, 12])), 2)
call_2221 = relay.TupleGetItem(func_445_call(relay.reshape(const_2220.astype('float32'), [1, 12])), 2)
output = relay.Tuple([call_2187,const_2188,bop_2197,bop_2209,call_2219,const_2220,])
output2 = relay.Tuple([call_2189,const_2188,bop_2200,bop_2212,call_2221,const_2220,])
func_2227 = relay.Function([], output)
mod['func_2227'] = func_2227
mod = relay.transform.InferType()(mod)
mutated_mod['func_2227'] = func_2227
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2227_call = mutated_mod.get_global_var('func_2227')
call_2228 = func_2227_call()
output = call_2228
func_2229 = relay.Function([], output)
mutated_mod['func_2229'] = func_2229
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2243 = relay.var("var_2243", dtype = "float64", shape = (9, 13))#candidate|2243|(9, 13)|var|float64
uop_2244 = relay.log2(var_2243.astype('float64')) # shape=(9, 13)
output = uop_2244
output2 = uop_2244
F = relay.Function([var_2243,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_2243,], output2)
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
	relay.transform.SimplifyExpr(),
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
input_2243= np.array([[4.718288,5.376726,0.473051,2.101387,8.220898,9.622972,-8.920083,0.694365,-6.222235,-6.874831,-2.589156,-2.219442,9.919411],[5.096252,-6.115977,6.356331,6.194789,-9.951970,9.606119,-0.144714,7.173391,-5.695786,-7.021711,7.121141,-3.916011,-8.951197],[4.564304,-8.965558,4.898574,-4.999125,1.359936,-2.969761,-4.814852,2.593931,-8.839046,6.913457,4.437701,3.885904,4.954607],[-5.631442,-2.111094,2.137308,1.650455,-8.329550,2.733160,-4.220197,6.345157,-5.813367,-3.788982,7.500274,5.167745,-3.900534],[-7.103835,-7.245173,-2.193685,-7.312449,3.935609,-7.750637,3.022911,-0.799703,4.948602,4.960882,9.471615,4.782417,1.742333],[-6.037156,9.086046,8.971394,2.102284,7.517928,6.368961,-0.659578,9.806667,3.276546,-6.116970,-5.509473,4.450397,-4.248080],[7.130743,-4.872971,-0.273902,5.286219,8.535114,-8.658404,4.404026,-3.099443,9.993355,5.404030,1.503111,7.390820,-1.492859],[-1.440858,-8.737625,-8.462776,-9.199801,-3.481465,-2.143263,-6.865007,0.346207,-0.018442,-5.436283,9.783355,0.421083,2.491157],[1.603006,-6.470299,-4.904151,0.203922,6.708243,-9.504362,-7.494216,9.062914,-6.174799,-5.840055,-5.785057,-9.359300,1.075846]], dtype='float64')
module1.set_input('var_2243', input_2243)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_2243, )
res3 = intrp3.evaluate()(input_2243, )
res4 = intrp4.evaluate()(input_2243, )
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
module5.set_input('var_2243', input_2243)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_2243, )
res7 = intrp7.evaluate()(input_2243, )
res8 = intrp8.evaluate()(input_2243, )
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
module9.set_input('var_2243', input_2243)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_2243, )
res11 = intrp11.evaluate()(input_2243, )
res12 = intrp12.evaluate()(input_2243, )
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
module13.set_input('var_2243', input_2243)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_2243, )
res15 = intrp15.evaluate()(input_2243, )
res16 = intrp16.evaluate()(input_2243, )
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
module17.set_input('var_2243', input_2243)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_2243, )
res19 = intrp19.evaluate()(input_2243, )
res20 = intrp20.evaluate()(input_2243, )
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
module21.set_input('var_2243', input_2243)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_2243, )
res23 = intrp23.evaluate()(input_2243, )
res24 = intrp24.evaluate()(input_2243, )
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