from tvm.relay.analysis import Feature
from functools import wraps
from tvm import relay
import pytest
from tvm.relay.ty import RefType
from tvm.relay.testing import Prelude
import sys
import logging
from tvm.relay.testing import enabled_targets
from tvm.relay.ty import TypeRelation
from tvm.relay import TypeFunctor
from tvm.relay.testing import create_workload
from tvm.relay import testing
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.op.annotation import compiler_end
from tvm.contrib.nvcc import have_fp16
import random
import math
from tvm.autotvm.tuner import RandomTuner
from tvm.relay.testing.synthetic import get_workload
from tvm.relay import TypeMutator
from tvm.runtime import container
from tvm.relay.testing import run_opt_pass
from tvm import te
from tvm.ir import IRModule
from tvm.relay.transform import un_cps
import numpy as np
from tvm.relay.transform import SimplifyInference
import tvm.relay.transform as _transform
from tvm.relay.analysis import detect_feature
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.ty import IncompleteType
from tvm.relay.ty import FuncType
from tvm.relay.build_module import bind_params_by_name
from tvm import topi
from numpy import isclose
import itertools
from tvm.relay import analysis
from tvm.relay.testing import make_nat_expr
from tvm.relay.adt import TypeData
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay import Any
from tvm import runtime
import tvm.relay.testing
from scipy import special
from tvm.relay import expr as _expr
from tvm.ir import structural_equal
from tvm.relay.scope_builder import ScopeBuilder
import json
from tvm.relay.ty import TensorType
from tvm import autotvm
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.analysis import get_calibration_data
from tvm.testing import assert_allclose
from tvm.relay import ExprVisitor
from tvm.relay.transform import FastMath
from tvm.relay.ty import TypeVar
from tvm.relay.data_dep_optimization import simplify_fc_transpose
import os
from tvm.relay.testing import check_grad
from tvm.relay.transform import to_cps
from tvm.relay.testing import rand
from tvm.relay.testing import run_infer_type
import scipy.sparse as sp
import scipy
import tvm.topi.testing
from typing import Union
from tvm.relay.ty import TupleType
from tvm.relay.testing import count
from tvm.relay.analysis import well_formed
import tvm
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.analysis import check_kind
from tvm.relay.prelude import Prelude
import tvm.relay as relay
from tvm.relay import op
from tvm.contrib import graph_runtime
import tvm.relay.transform
from tvm.relay import TypeVisitor
from tvm.relay import transform
import tvm.testing
from tvm.relay.ty import TypeCall
from tvm import relay as rly
import time
from tvm.relay import create_executor
from tvm import nd

fau5e=relay.scalar_type('''int16''')
Ejpxq=relay.const(0,fau5e)
O732T=relay.var('''data''',fau5e)
kA0Jc=relay.equal(O732T,Ejpxq)
Ow3eD=relay.ScopeBuilder()
gwXBf=Ow3eD.if_scope(kA0Jc)
with gwXBf:
	Ow3eD.ret(O732T)

i7XFd=relay.var('''x''',dtype='''bool''',shape=())
Q5kAj=relay.qnn.op.concatenate((i7XFd,i7XFd,),input_scales=(Ejpxq,Ejpxq,),input_zero_points=(Ejpxq,Ejpxq,),output_scale=Ejpxq,output_zero_point=Ejpxq,axis=0)
O5ldS=relay.nn.conv2d(i7XFd,i7XFd,kernel_size=(3,1,),padding=(1,1,),groups=32)
Yy1Gs=relay.add(O5ldS,O5ldS)
XlRVy=relay.Function([i7XFd,],Yy1Gs)
HAB5X=tvm.IRModule()
S4DRV=relay.GlobalVar('''f''')
iyOf3=XlRVy.with_attr('''Compiler''','''test_graph''')
HAB5X[S4DRV]=iyOf3
HAB5X[S4DRV]=XlRVy


def run_infer_type(expr, mod=None):
    if (not mod):
        mod = tvm.IRModule.from_expr(expr)
        mod = transform.InferType()(mod)
        entry = mod['main']
        return (entry if isinstance(expr, relay.Function) else entry.body)
    else:
        if isinstance(expr, relay.GlobalVar):
            gv = expr.name_hint
        else:
            func = expr
            if (not isinstance(expr, relay.Function)):
                func = relay.Function(analysis.free_vars(expr), expr)
            mod['main'] = func
            gv = 'main'
        mod = transform.InferType()(mod)
        if isinstance(expr, (relay.GlobalVar, relay.Function)):
            return mod[gv]
        return mod[gv].body
pcfeh=relay.TypeVar('''b''')
GDWr5=relay.Var('''f''',pcfeh)
UNK8i=relay.Function([GDWr5,],GDWr5,pcfeh,[pcfeh,])
PKmcC=run_infer_type(UNK8i)
HAB5X['''main''']=PKmcC
PQTK9=XlRVy.with_attr('''Compiler''','''test_graph''')
HAB5X[S4DRV]=PQTK9
HAB5X['''main''']=XlRVy
HAB5X['''main''']=XlRVy
HAB5X['''main''']=XlRVy


def set_external_func_attr(func, compiler, ext_symbol):
    func = func.with_attr('Primitive', tvm.tir.IntImm('int32', 1))
    func = func.with_attr('Compiler', compiler)
    func = func.with_attr('global_symbol', ext_symbol)
    return func
kVNuS=set_external_func_attr(XlRVy,'''dnnl''','''dnnl_0''')
Caq25=relay.Call(kVNuS,[i7XFd,i7XFd,i7XFd,])
tyITD=tvm.IRModule.from_expr(XlRVy)
tMj9Z=relay.qnn.transform.CanonicalizeOps()
NcJuu=tMj9Z(tyITD)
oZaed=tvm.cpu(0)
wBx1i=relay.create_executor('''graph''',ctx=oZaed,target='''llvm''')
luSiC=wBx1i.evaluate(NcJuu['''main'''])
JbPW6=np.arange((-32),32,1)
GZJcF=JbPW6.reshape(0,64)
N2eZ3=GZJcF.astype('''uint16''')
Ycp7F=JbPW6.reshape(1,8)
S0xIh=Ycp7F.astype('''float64''')
Ct8xa=luSiC(N2eZ3,S0xIh)
SBKST=Ct8xa.asnumpy()
RqKj3=np.concatenate((N2eZ3,S0xIh,),axis=0)
np.testing.assert_equal(SBKST,RqKj3)
YMN4h=relay.Function([i7XFd,],Yy1Gs)
EKb6T=relay.var('''shared''')
ALi1w=relay.Let(EKb6T,i7XFd,GDWr5)
check_basic_block_normal_form(ALi1w)
g9e4d=transform.LazyGradientInit()
nwg51=g9e4d(HAB5X)
vJx7d=create_executor(mod=nwg51)
HAB5X['''main''']=YMN4h
lMRC2=vJx7d.evaluate(HAB5X['''main'''])
BJ4pw=rand('''uint64''',*(15,11,))
Dq7kJ=lMRC2(BJ4pw)
y1EtJ=Dq7kJ.asnumpy()
QMUsC=BJ4pw.asnumpy()
assert_allclose(y1EtJ,(QMUsC * QMUsC))
psAog=BJ4pw.asnumpy()
vxe7d=BJ4pw.asnumpy()
AvDfl=lMRC2(psAog,vxe7d)
MAeBP=AvDfl.asnumpy()
assert_allclose(MAeBP,(QMUsC * QMUsC))
CZ0DO=relay.If(i7XFd,EKb6T,Yy1Gs)
dzWxG=relay.TupleGetItem(Caq25,0)
PQ4t1=YMN4h.with_attr('''Compiler''','''test_graph''')
