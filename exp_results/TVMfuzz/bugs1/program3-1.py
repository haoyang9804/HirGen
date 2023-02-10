from tvm.relay.analysis import well_formed
from tvm.relay.ty import FuncType
from tvm import relay as rly
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.testing import create_workload
from tvm.relay.transform import to_cps
from tvm.relay.transform import SimplifyInference
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import make_nat_expr
from tvm.relay.testing import Prelude
import pytest
from tvm.relay.ty import RefType
import tvm.relay.testing
from tvm.ir import IRModule
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.prelude import Prelude
from tvm.relay import testing
from tvm.relay.transform import un_cps
from tvm.relay.testing import run_infer_type
from functools import wraps
from tvm.relay.op.annotation import compiler_end
from tvm.relay.backend.interpreter import ConstructorValue
import random
import tvm.relay as relay
from tvm import runtime
from tvm.relay.ty import TypeVar
from tvm import topi
from tvm.runtime import container
from tvm.relay.testing import count
from tvm.relay import create_executor
from tvm.contrib.nvcc import have_fp16
import scipy
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.testing import assert_allclose
from tvm.relay.testing import run_opt_pass
from tvm.relay.analysis import get_calibration_data
from tvm.relay import TypeFunctor
from tvm.relay import TypeVisitor
from tvm.relay import expr as _expr
from tvm.relay.analysis import Feature
import tvm.testing
from numpy import isclose
from tvm.relay.ty import IncompleteType
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.adt import TypeData
from tvm.relay.ty import TensorType
from tvm import relay
from tvm import autotvm
import tvm.relay.transform
import time
from tvm.ir import structural_equal
from tvm.relay.testing import check_grad
from tvm import te
import os
from tvm.relay.ty import TupleType
import sys
from tvm.relay import analysis
import itertools
import tvm
from tvm.relay.analysis import detect_feature
from tvm.relay.testing import rand
import scipy.sparse as sp
from tvm.relay.testing import enabled_targets
from tvm.relay.ty import TypeRelation
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay import op
from tvm.relay import TypeMutator
from tvm.relay.ty import GlobalTypeVar
import numpy as np
from tvm.autotvm.tuner import RandomTuner
from typing import Union
from tvm.relay.analysis import check_kind
from tvm.relay.transform import FastMath
import tvm.topi.testing
import json
from tvm import nd
import tvm.relay.transform as _transform
from tvm.relay import ExprVisitor
from tvm.relay.ty import TypeCall
import math
from tvm.relay import transform
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay import Any
from tvm.contrib import graph_runtime
from scipy import special
import logging

LkIrG=relay.var('''weight_conv''',shape=(1,15,))
FMze1=relay.scalar_type('''float16''')
ukTlz=relay.Var('''d''',FMze1)
b0cXM=relay.const(0.0,'''float64''')
INAEA=relay.If(ukTlz,LkIrG,b0cXM)
zfj03=relay.nn.conv2d(LkIrG,LkIrG,channels=64,kernel_size=(3,3,),padding=(1,0,))
xjwcu=relay.add(INAEA,zfj03)
lkrbC=relay.Function([LkIrG,LkIrG,LkIrG,],xjwcu)


def get_recursive_count_loop():
    mod = tvm.IRModule({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    sb = relay.ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, dtype='int32'))):
        sb.ret(i)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, dtype='int32'))
        rec_call = relay.Call(sum_up, [one_less])
        sb.ret(relay.add(rec_call, i))
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], 'int32'))
    func = func.with_attr('Inline', tvm.tir.IntImm('int32', 1))
    mod[sum_up] = func
    iarg = relay.var('i', shape=[], dtype='int32')
    mod['main'] = relay.Function([iarg], sum_up(iarg))
    return (mod, sum_up)
Ow4nn=get_recursive_count_loop()
Ow4nn[0]['''main''']=lkrbC
Ipaxr=tvm.IRModule({})
V9WoA=relay.GlobalVar('''sum_up''')
mBPl7=relay.var('''i''',shape=[],dtype='''uint1''')
w37EY=relay.ScopeBuilder()
UvAxe=w37EY.get()
Iqmww=relay.TensorType((1,2,),'''int16''')
sUex0=relay.Function([mBPl7,],UvAxe,ret_type=Iqmww)
xHQOZ=sUex0.with_attr('''Compiler''','''a''')
Ipaxr[V9WoA]=xHQOZ
Ipaxr[V9WoA]=lkrbC
Ipaxr['''main''']=lkrbC
mqN3w=tvm.IRModule()
mqN3w['''main''']=lkrbC
rFD5y=to_cps(mqN3w['''main'''],mod=mqN3w)
mqN3w['''main''']=rFD5y
mqN3w['''main''']=rFD5y
n2Amx=un_cps(mqN3w['''main'''])
mqN3w['''main''']=n2Amx
Bx6Z0=relay.GlobalTypeVar('''Ayy''')
jrYuC=relay.TypeData(Bx6Z0,[],[])
mqN3w[Bx6Z0]=jrYuC
mqN3w[V9WoA]=lkrbC
l9gIp=relay.RefCreate(lkrbC)
MvxG7=relay.Var('''x''')
v0OJs=relay.Tuple([])
IpFnL=relay.Let(MvxG7,l9gIp,v0OJs)

SEMVER = '#[version = "0.0.5"]\n'

BINARY_OPS = {'*': relay.multiply, '/': relay.divide, '+': relay.add, '-': relay.subtract, '<': relay.less, '>': relay.greater, '<=': relay.less_equal, '>=': relay.greater_equal, '==': relay.equal, '!=': relay.not_equal}

TYPES = {'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64', 'bool', 'int8x4', 'uint1x4', 'float16x4'}

LIST_DEFN = '\ntype List[A] {\n    Cons(A, List[A]),\n    Nil,\n}\n'


def assert_graph_equal(lhs, rhs):
    tvm.ir.assert_structural_equal(lhs, rhs, map_free_vars=True)


def graph_equal(lhs, rhs):
    return tvm.ir.structural_equal(lhs, rhs, map_free_vars=True)


def roundtrip_expr(expr):
    text = tvm.relay.Expr.astext(expr, show_meta_data=False)
    x = tvm.parser.parse_expr(text)
    assert_graph_equal(x, expr)


def roundtrip(expr):
    x = tvm.parser.fromtext(expr.astext())
    assert_graph_equal(x, expr)


def parse_text(code):
    expr = tvm.parser.parse_expr(code)
    roundtrip_expr(expr)
    return expr


def parses_as(code, expr):
    parsed = parse_text(code)
    result = graph_equal(parsed, expr)
    return result


def parse_module(code):
    mod = tvm.parser.parse((SEMVER + code))
    roundtrip(mod)
    return mod


def assert_parses_as(code, expr):
    parsed = parse_text(code)
    assert_graph_equal(parsed, expr)
assert_parses_as('''let %x = 1; ()''',IpFnL)
