from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.transform import FastMath
from tvm.relay.testing import create_workload
from tvm.relay.analysis import detect_feature
import scipy
from tvm.relay import TypeFunctor
import random
from scipy import special
from tvm.relay.testing.temp_op_attr import TempOpAttr
import tvm.relay as relay
from tvm.relay.backend.interpreter import RefValue
from tvm.runtime import container
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.ty import TypeRelation
from tvm.relay.testing import check_grad
from tvm.relay.ty import RefType
from tvm.contrib import graph_runtime
from tvm.testing import assert_allclose
from tvm.relay.testing import run_opt_pass
from tvm.relay.ty import GlobalTypeVar
from tvm.ir import structural_equal
from tvm import nd
from typing import Union
import tvm.relay.transform as _transform
from tvm.relay.op.annotation import compiler_end
import tvm
from tvm import relay
from tvm.relay.analysis import check_basic_block_normal_form
import tvm.topi.testing
from tvm.relay import ExprVisitor
from tvm.relay.analysis import check_kind
from tvm.relay.ty import TupleType
from tvm.relay.transform import to_cps
from numpy import isclose
import os
from tvm.relay.transform import un_cps
from tvm.relay.analysis import get_calibration_data
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay import analysis
from tvm.relay.testing import run_infer_type
from tvm.relay.scope_builder import ScopeBuilder
import scipy.sparse as sp
from tvm.relay.testing import rand
from tvm import relay as rly
from tvm.relay.ty import TypeCall
import logging
import tvm.relay.transform
from tvm.relay import testing
from tvm import te
from tvm.relay.testing import count
from functools import wraps
from tvm.relay import expr as _expr
from tvm.contrib.nvcc import have_fp16
from tvm.relay import Any
from tvm.relay.op.annotation import compiler_begin
import sys
from tvm.relay.analysis import Feature
from tvm.ir import IRModule
from tvm import runtime
from tvm.relay.testing import make_nat_expr
from tvm.relay.ty import TypeVar
import json
from tvm.relay.analysis import well_formed
import itertools
from tvm import topi
import math
from tvm.relay import op
from tvm.relay import create_executor
import tvm.testing
from tvm.relay.testing import Prelude
import tvm.relay.testing
from tvm.relay.prelude import Prelude
from tvm.relay.transform import SimplifyInference
from tvm.relay.testing import enabled_targets
from tvm.relay import TypeVisitor
from tvm.relay.adt import TypeData
import time
from tvm.relay.ty import FuncType
from tvm.relay.ty import IncompleteType
from tvm.autotvm.tuner import RandomTuner
from tvm.relay.backend.interpreter import ConstructorValue
from tvm import autotvm
import numpy as np
import pytest
from tvm.relay import TypeMutator
from tvm.relay.ty import TensorType

DhXi0=relay.TypeVar('''A''')
Oeh3e=relay.GlobalTypeVar('''T''')
q5wh9=relay.TypeData(Oeh3e,[DhXi0,],[])
Fbs3v=tvm.IRModule()
QOS1H=relay.FuncType([DhXi0,],DhXi0)
IqTGz=relay.var('''w''',QOS1H)
XQhKm=relay.var('''x''',dtype='''float16''',shape=(16,8,))
eWy3C=relay.Var('''True''',QOS1H)
PzgQc=relay.const(1,dtype='''int64''')
AaLng=relay.If(XQhKm,eWy3C,PzgQc)
YxiMx=relay.Function([IqTGz,IqTGz,IqTGz,],AaLng)
Fbs3v['''main''']=YxiMx
Fbs3v['''main''']=YxiMx
Fbs3v[Oeh3e]=q5wh9


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


def assert_has_type(expr, typ, mod=tvm.IRModule({})):
    checked_expr = run_infer_type(expr, mod)
    checked_type = checked_expr.checked_type
    if (checked_type != typ):
        raise RuntimeError(('Type mismatch %s vs %s' % (checked_type, typ)))


def initialize_box_adt(mod):
    box = relay.GlobalTypeVar('box')
    tv = relay.TypeVar('tv')
    constructor = relay.Constructor('constructor', [tv], box)
    data = relay.TypeData(box, [tv], [constructor])
    mod[box] = data
    return (box, constructor)
XXcpb=initialize_box_adt(Fbs3v)
PnZGc=XXcpb[0](DhXi0)
F8zZo=relay.FuncType([DhXi0,],PnZGc,[DhXi0,])


def run_opt_pass(expr, passes):
    passes = (passes if isinstance(passes, list) else [passes])
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod['main']
    return (entry if isinstance(expr, relay.Function) else entry.body)


def check_eval(expr, expected_result, mod=None, rtol=1e-07):
    ctx = tvm.context('llvm', 0)
    intrp = create_executor(mod=mod, ctx=ctx, target='llvm')
    result = intrp.evaluate(expr)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)


def test_no_explicit_bind():
    x = relay.const(1)
    y = op.add(x, x)
    z = op.add(y, y)
    f = relay.Function([], op.add(z, z))
    '\n    fn () {\n      %0 = add(1, 1);\n      %1 = add(%0, %0);\n      add(%1, %1)\n    }\n    '
    assert (not (Feature.fLet in detect_feature(f)))
    bblock = run_opt_pass(f, transform.ToBasicBlockNormalForm())
    assert (Feature.fLet not in detect_feature(bblock))
    check_eval(f(), 8.0)
    check_eval(bblock(), 8.0)
    check_basic_block_normal_form(bblock)


def test_top_level_nested_if():
    x = relay.var('x', shape=(), dtype='bool')
    y = relay.var('y', shape=(), dtype='float32')
    z = relay.var('z', shape=(), dtype='float32')
    cond_t = relay.const(True)
    cond_f = relay.const(False)
    one = relay.const(1, dtype='float32')
    three = relay.const(3, dtype='float32')
    y2 = relay.add(y, y)
    z2 = relay.add(z, z)
    true_branch = relay.If(cond_t, relay.add(z2, y2), relay.add(three, y2))
    false_branch = relay.If(cond_f, z2, one)
    body = relay.If(x, true_branch, false_branch)
    '\n    free_var %x: bool\n    if (%x) {\n      if (True) {\n        free_var %z: float32\n        %0 = add(%z, %z);\n        free_var %y: float32\n        %1 = add(%y, %y);\n        add(%0, %1)\n      } else {\n        add(3f, %1)\n      }\n    } else {\n      if (False) {\n        %0\n      } else {\n        1f\n      }\n    }\n    '

    def expected():
        x = relay.var('x', shape=(), dtype='bool')
        y = relay.var('y', shape=(), dtype='float32')
        z = relay.var('z', shape=(), dtype='float32')
        cond_t = relay.const(True)
        cond_f = relay.const(False)
        one = relay.const(1, dtype='float32')
        three = relay.const(3, dtype='float32')
        y2 = relay.var('y2')
        z2 = relay.var('z2')
        true_branch = relay.If(cond_t, relay.add(z2, y2), relay.add(three, y2))
        true_branch = relay.Let(y2, relay.add(y, y), true_branch)
        false_branch = relay.If(cond_f, z2, one)
        body = relay.If(x, true_branch, false_branch)
        body = relay.Let(z2, relay.add(z, z), body)
        return body
    bblock = run_opt_pass(body, [transform.ToBasicBlockNormalForm()])
    '\n    free_var %z: float32\n    let %x: float32 = add(%z, %z) /* ty=float32 */;\n    free_var %x1: bool\n    if (%x1) {\n      free_var %y: float32\n      let %x2: float32 = add(%y, %y) /* ty=float32 */;\n      if (True /* ty=bool */) {\n        add(%x, %x2) /* ty=float32 */\n      } else {\n        add(3f /* ty=float32 */, %x2) /* ty=float32 */\n      }\n    } else {\n      if (False /* ty=bool */) {\n        %x\n      } else {\n        1f /* ty=float32 */\n      }\n    }\n    '
    expected_output = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(bblock, expected_output, map_free_vars=True)

x = relay.var('x', shape=(), dtype='bool')

y = relay.var('y', shape=(), dtype='float32')

z = relay.var('z', shape=(), dtype='float32')

cond_t = relay.const(True)

cond_f = relay.const(False)

one = relay.const(1, dtype='float32')

three = relay.const(3, dtype='float32')

y2 = relay.add(y, y)

z2 = relay.add(z, z)

true_branch = relay.If(cond_t, relay.add(z2, y2), relay.add(three, y2))

false_branch = relay.If(cond_f, z2, one)

body = relay.If(x, true_branch, false_branch)


def expected():
    x = relay.var('x', shape=(), dtype='bool')
    y = relay.var('y', shape=(), dtype='float32')
    z = relay.var('z', shape=(), dtype='float32')
    cond_t = relay.const(True)
    cond_f = relay.const(False)
    one = relay.const(1, dtype='float32')
    three = relay.const(3, dtype='float32')
    y2 = relay.var('y2')
    z2 = relay.var('z2')
    true_branch = relay.If(cond_t, relay.add(z2, y2), relay.add(three, y2))
    true_branch = relay.Let(y2, relay.add(y, y), true_branch)
    false_branch = relay.If(cond_f, z2, one)
    body = relay.If(x, true_branch, false_branch)
    body = relay.Let(z2, relay.add(z, z), body)
    return body
VC6Dc=F8zZo()


def run_opt_pass(expr, passes):
    passes = (passes if isinstance(passes, list) else [passes])
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod['main']
    return (entry if isinstance(expr, relay.Function) else entry.body)
I5BXx=transform.InferType()
eTUvu=run_opt_pass(YxiMx,I5BXx)


def run_opt_pass(expr, passes):
    passes = (passes if isinstance(passes, list) else [passes])
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod['main']
    return (entry if isinstance(expr, relay.Function) else entry.body)


def check_eval(expr, expected_result, mod=None, rtol=1e-07):
    ctx = tvm.context('llvm', 0)
    intrp = create_executor(mod=mod, ctx=ctx, target='llvm')
    result = intrp.evaluate(expr)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)
lk4Ha=IqTGz(PzgQc)
check_eval(lk4Ha,3)
cHmYZ=eTUvu(PzgQc)


def run_opt_pass(expr, passes):
    passes = (passes if isinstance(passes, list) else [passes])
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod['main']
    return (entry if isinstance(expr, relay.Function) else entry.body)


def check_eval(expr, expected_result, mod=None, rtol=1e-07):
    ctx = tvm.context('llvm', 0)
    intrp = create_executor(mod=mod, ctx=ctx, target='llvm')
    result = intrp.evaluate(expr)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)
check_eval(eTUvu,2)
check_basic_block_normal_form(eTUvu)
t7qYN=XXcpb[1](eWy3C)
VUP7P=relay.Function([IqTGz,IqTGz,],t7qYN,DhXi0,[DhXi0,])


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
pFwKG=run_infer_type(VUP7P,Fbs3v)
NEucZ=relay.Var('''curried_mult''')
AN8lY=relay.RefRead(NEucZ)
dQlXO=relay.const(2)
RI8dl=relay.multiply(dQlXO,NEucZ)
jveHg=relay.Let(NEucZ,AN8lY,RI8dl)

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
Lpas4=relay.Call(NEucZ,[],None,None)
assert_parses_as('''
        (fn (%x) { %x })(0)
        ''',Lpas4)


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
W2l0I=run_infer_type(VUP7P)
P4A2z=Any()
ndjqX=relay.TensorType([P4A2z,1,],dtype='''float32''')
tvm.ir.assert_structural_equal(W2l0I.ret_type,ndjqX)
yYMiB=to_cps(W2l0I)
