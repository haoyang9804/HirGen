import scipy.sparse as sp
from tvm.relay.ty import IncompleteType
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm import relay
from tvm.relay import transform
from tvm.relay import TypeVisitor
from tvm.ir import IRModule
import tvm.relay.transform
from tvm.relay.ty import RefType
from tvm.relay.testing import check_grad
from tvm.relay.analysis import well_formed
from tvm.relay.ty import TypeVar
import json
from functools import wraps
import itertools
from tvm.relay.testing import enabled_targets
from scipy import special
import time
from tvm.relay.testing import Prelude
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.ty import TupleType
import tvm.topi.testing
from tvm.contrib.nvcc import have_fp16
from tvm.autotvm.tuner import RandomTuner
from tvm import nd
import tvm.relay.transform as _transform
from tvm.relay import TypeFunctor
from tvm import autotvm
from tvm.relay import analysis
from tvm.relay.analysis import detect_feature
from tvm.relay import TypeMutator
from tvm.relay.adt import TypeData
from tvm.relay import create_executor
from tvm.relay.backend.interpreter import ConstructorValue
from tvm import runtime
from tvm import topi
from tvm.contrib import graph_runtime
import numpy as np
from tvm.relay.op.annotation import compiler_end
import pytest
from tvm.relay.testing import create_workload
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.testing import run_infer_type
from tvm import relay as rly
from tvm.relay.testing import make_nat_expr
import os
from tvm.ir import structural_equal
from tvm.relay.testing import count
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import rand
from tvm.relay.analysis import Feature
from tvm.relay.testing import run_opt_pass
import tvm.relay.testing
from tvm.relay.ty import TypeCall
from tvm.relay.analysis import check_kind
import tvm.relay as relay
from tvm.relay import Any
from tvm.relay.transform import FastMath
from tvm.relay.transform import SimplifyInference
import tvm
from tvm import te
import scipy
from tvm.relay.transform import to_cps
import logging
from typing import Union
from tvm.relay import ExprVisitor
from tvm.relay import op
import random
from tvm.relay.ty import TensorType
import sys
from tvm.relay.testing.synthetic import get_workload
from tvm.relay import expr as _expr
from tvm.runtime import container
from tvm.relay.scope_builder import ScopeBuilder
import math
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.transform import un_cps
from tvm.testing import assert_allclose
import tvm.testing
from tvm.relay.analysis import check_basic_block_normal_form
from numpy import isclose
from tvm.relay.prelude import Prelude
from tvm.relay.analysis import get_calibration_data
from tvm.relay.ty import FuncType
from tvm.relay.ty import TypeRelation
from tvm.relay import testing



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
bMxZ4=relay.var('''cond''',shape=(),dtype='''uint16''')
M5BXc=np.ones((10,2,),'''uint32''')
j7OHu=relay.const(M5BXc)
NBo3o=relay.Function([bMxZ4,],j7OHu)
TDmdW=tvm.IRModule({})
FVLXc=relay.GlobalVar('''sum_up''')
TDmdW[FVLXc]=NBo3o
TDmdW['''main''']=NBo3o
sRuJ6=run_infer_type(NBo3o)
kPygK=relay.transform.gradient(sRuJ6)


def test_id():
    x = relay.var('x', shape=[])
    id = run_infer_type(relay.Function([x], x))
    id_cps = run_infer_type(to_cps(id))


def test_double():
    t = relay.TypeVar('t')
    x = relay.var('x', t)
    f = relay.var('f', relay.FuncType([t], t))
    double = run_infer_type(relay.Function([f, x], f(f(x)), t, [t]))
    double_cps = run_infer_type(to_cps(double))


def test_recursion():
    mod = tvm.IRModule()
    p = Prelude(mod)
    shape = (10, 10)
    dtype = 'float32'
    t = relay.TensorType(shape, dtype)
    x = relay.var('x', t)
    double = relay.Function([x], (x + x))
    i = relay.var('i', t)
    func = relay.Function([i], p.nat_iterate(double, make_nat_expr(p, 3))(i))
    mod['main'] = func
    mod['main'] = to_cps(mod['main'], mod=mod)
    mod['main'] = un_cps(mod['main'])
    ex = create_executor(mod=mod)
    i_nd = rand(dtype, *shape)
    forward = ex.evaluate()(i_nd)
    tvm.testing.assert_allclose(forward.asnumpy(), (8 * i_nd.asnumpy()))


def test_cps_pe():

    def destroy_ref(x):
        x = run_infer_type(x)
        x = to_cps(x)
        x = run_infer_type(x)
        y = un_cps(x)
        y = run_infer_type(y)
        x = run_opt_pass(x, tvm.transform.Sequential([transform.PartialEvaluate(), transform.DeadCodeElimination(inline_once=True)]))
        assert (Feature.fRefCreate not in detect_feature(x))
    unit = relay.Function([], relay.const(0.0, dtype='float32'))
    f_ref = relay.Var('f_ref')
    one = relay.const(1.0, dtype='float32')
    two = relay.const(2.0, dtype='float32')
    cond = relay.var(shape=(), dtype='uint1', name_hint='cond')
    true_branch = relay.RefWrite(f_ref, relay.Function([], one))
    false_branch = relay.RefWrite(f_ref, relay.Function([], two))
    if_expr = relay.If(cond, true_branch, false_branch)
    stmt = relay.Let(f_ref, relay.RefCreate(unit), relay.Let(relay.Var('x'), if_expr, relay.Call(relay.RefRead(f_ref), [])))
    F = relay.Function([cond], stmt)
    destroy_ref(F)
    G = relay.Function([cond], relay.If(cond, one, two))
    G = run_infer_type(G)
    G = relay.transform.gradient(G)
    destroy_ref(G)
    x = relay.var('x', shape=(1, 16))
    y = relay.var('y', shape=(1, 16))
    z = relay.var('z', shape=(1, 16))
    cond = relay.var('cond', shape=(), dtype='uint1')
    H = relay.If(cond, x, y)
    H = relay.add(H, z)
    H = relay.Function([cond, x, y, z], H)
    H = run_infer_type(H)
    H = relay.transform.gradient(H)
    destroy_ref(H)


def destroy_ref(x):
    x = run_infer_type(x)
    x = to_cps(x)
    x = run_infer_type(x)
    y = un_cps(x)
    y = run_infer_type(y)
    x = run_opt_pass(x, tvm.transform.Sequential([transform.PartialEvaluate(), transform.DeadCodeElimination(inline_once=True)]))
    assert (Feature.fRefCreate not in detect_feature(x))
destroy_ref(kPygK)
