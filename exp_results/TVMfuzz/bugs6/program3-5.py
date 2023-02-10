from tvm.relay.ty import TypeRelation
from tvm.relay import expr as _expr
from tvm.relay.testing import enabled_targets
import tvm.testing
from numpy import isclose
from tvm.relay import ExprVisitor
from tvm.relay.testing import make_nat_expr
from tvm.relay.op.annotation import compiler_begin
import tvm.relay.transform
from tvm.relay.testing.synthetic import get_workload
from tvm import nd
import logging
from tvm.runtime import container
from tvm.relay.analysis import check_kind
from tvm.relay.backend.interpreter import RefValue
from tvm.relay import create_executor
from tvm.autotvm.tuner import RandomTuner
import itertools
from tvm.relay.testing import run_infer_type
from tvm.relay.ty import TypeVar
from tvm.relay.ty import TupleType
from tvm.relay.backend.interpreter import ConstructorValue
from functools import wraps
from tvm import relay as rly
from tvm.relay import testing
from tvm import topi
from tvm.relay.analysis import well_formed
from tvm import runtime
from scipy import special
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.testing import check_grad
from tvm.relay import TypeFunctor
from tvm.relay.analysis import Feature
from typing import Union
import json
from tvm.relay.op.annotation import compiler_end
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.data_dep_optimization import simplify_fc_transpose
import sys
from tvm.relay import TypeMutator
import scipy
from tvm.relay.ty import RefType
from tvm.relay import analysis
from tvm.relay.ty import IncompleteType
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.transform import FastMath
from tvm.contrib import graph_runtime
import tvm
import random
from tvm.relay.transform import to_cps
from tvm.contrib.nvcc import have_fp16
from tvm.relay import op
from tvm.relay.analysis import detect_feature
from tvm.relay.adt import TypeData
from tvm.relay.ty import TypeCall
import scipy.sparse as sp
from tvm.relay.testing import Prelude
from tvm.relay.analysis import get_calibration_data
from tvm.relay.ty import TensorType
from tvm.relay.ty import FuncType
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass
from tvm.ir import IRModule
import os
import tvm.relay as relay
from tvm.relay.prelude import Prelude
from tvm.ir import structural_equal
import math
import time
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay.testing import create_workload
from tvm import relay
import pytest
import tvm.topi.testing
import numpy as np
from tvm.relay.testing import rand
from tvm.relay.testing import count
import tvm.relay.transform as _transform
from tvm.relay import TypeVisitor
from tvm.relay.transform import SimplifyInference
from tvm import te
from tvm.relay import Any
from tvm import autotvm
import tvm.relay.testing
from tvm.relay.transform import un_cps
from tvm.testing import assert_allclose
from tvm.relay.ty import GlobalTypeVar

q3QHc=tvm.gpu()
J4Bbs=relay.scalar_type('''float32''')
T0Nd4=relay.Var('''size''',J4Bbs)
DT8UA=relay.op.memory.alloc_storage(T0Nd4,T0Nd4,q3QHc)
HzXKI=relay.var('''w1''',shape=(),dtype='''int32''')
QxDT1=relay.const(2.0)
eIzYH=relay.Function([HzXKI,HzXKI,],QxDT1)
cRQMS=tvm.IRModule()
cRQMS['''main''']=eIzYH
cRQMS['''main''']=eIzYH
cRQMS['''main''']=eIzYH


def run_func(func, params, x):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, 'llvm', params=params)
    from tvm.contrib import graph_runtime
    ctx = tvm.cpu(0)
    dtype = 'float32'
    m = graph_runtime.GraphModule(lib['default'](ctx))
    m.set_input('data', tvm.nd.array(x.astype(dtype)))
    m.run()
    tvm_output = m.get_output(0)
    return tvm_output.asnumpy()
nkVSt=np.random.uniform(0,1,(29,1,3,3,))
eTBkh=nkVSt.astype('''uint32''')
YkRBQ=tvm.nd.array(eTBkh)
BfajC=np.random.randn(1,22)
cK3O4=BfajC.astype('''float64''')
qsBCc=run_func(eIzYH,{'''w1''':YkRBQ,'''w2''':YkRBQ},cK3O4)
np.testing.assert_allclose(qsBCc,qsBCc,atol=1e-05,rtol=1e-05)


def set_external_func_attr(func, compiler, ext_symbol):
    func = func.with_attr('Primitive', tvm.tir.IntImm('int32', 1))
    func = func.with_attr('Compiler', compiler)
    func = func.with_attr('global_symbol', ext_symbol)
    return func
v9Qzs=set_external_func_attr(eIzYH,'''dnnl''','''ccompiler_2''')
XekOj=relay.var('''x2''',shape=(0,2,))
acc8V=relay.Call(v9Qzs,[XekOj,XekOj,])
VzloB=tvm.IRModule.from_expr(acc8V)
SVlY3=relay.qnn.transform.CanonicalizeOps()
mjQFT=SVlY3(VzloB)
BS1Aj=tvm.cpu(0)
XvJMi=relay.create_executor('''graph''',ctx=BS1Aj,target='''llvm''')
COQww=XvJMi.evaluate(mjQFT['''main'''])
A1XNW=np.arange((-32),32,0)
zMVMf=A1XNW.reshape(1,10)
zVh48=zMVMf.astype('''float16''')
gL8yX=A1XNW.reshape(1,64)
j66EV=gL8yX.astype('''float64''')
gg2Ob=COQww(zVh48,j66EV)
iXOcV=gg2Ob.asnumpy()
OvYaO=np.concatenate(((zVh48 + 1),j66EV,),axis=0)
np.testing.assert_equal(iXOcV,OvYaO)
SbkpC=np.power(1,32)
vTnkD=relay.const(((62 + 42) / (SbkpC - 1.0)),'''uint64''')
zFEOO=eIzYH(vTnkD)


def run_opt_pass(expr, opt_pass):
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod['main']
    return (entry if isinstance(expr, relay.Function) else entry.body)


def check_eval(expr, args, expected_result, mod=None, rtol=1e-07):
    if (mod is None):
        mod = tvm.IRModule()
    ctx = tvm.context('llvm', 0)
    intrp = create_executor(mod=mod, ctx=ctx, target='llvm')
    result = intrp.evaluate(expr)(*args)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)
check_eval(zFEOO,8)


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
ACw7i=run_infer_type(eIzYH)
mIjq7=transform.LazyGradientInit()
Yl7lc=mIjq7(cRQMS)
iabQi=create_executor(mod=Yl7lc)
cRQMS['''main''']=ACw7i
AUBk9=iabQi.evaluate(cRQMS['''main'''])
lFEyh=rand('''float16''',*(10,10,))
INmnv=AUBk9(lFEyh)
WgRRF=simplify_fc_transpose.convert(eIzYH,{'''w1''':YkRBQ,'''w2''':YkRBQ})


def run_opt_pass(expr, opt_pass):
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod['main']
    return (entry if isinstance(expr, relay.Function) else entry.body)
VmNdx=transform.ToANormalForm()
znRwo=run_opt_pass(eIzYH,VmNdx)
lrayh=znRwo(vTnkD)


def run_opt_pass(expr, opt_pass):
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod['main']
    return (entry if isinstance(expr, relay.Function) else entry.body)


def check_eval(expr, args, expected_result, mod=None, rtol=1e-07):
    if (mod is None):
        mod = tvm.IRModule()
    ctx = tvm.context('llvm', 0)
    intrp = create_executor(mod=mod, ctx=ctx, target='llvm')
    result = intrp.evaluate(expr)(*args)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)
check_eval(lrayh,4)
cRQMS['''main''']=eIzYH
yqn0W=relay.TensorType((),'''float64''')
tvm.ir.assert_structural_equal(cRQMS['''main'''],[yqn0W,])
