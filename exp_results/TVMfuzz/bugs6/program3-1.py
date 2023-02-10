from tvm.relay.analysis import get_calibration_data
from numpy import isclose
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.transform import un_cps
from tvm.autotvm.tuner import RandomTuner
import random
from tvm.relay.ty import TypeVar
from tvm import nd
import sys
from tvm.relay.adt import TypeData
from tvm.relay.testing import enabled_targets
import tvm.relay as relay
from tvm.relay.testing import rand
from tvm.relay import TypeVisitor
from tvm.relay import expr as _expr
import logging
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.testing import assert_allclose
from functools import wraps
from tvm.relay.ty import IncompleteType
from tvm.relay.testing import make_nat_expr
from tvm.relay.op.annotation import compiler_end
import numpy as np
import os
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.analysis import check_basic_block_normal_form
import scipy
from tvm import autotvm
from tvm.relay.analysis import detect_feature
from tvm import relay as rly
import time
import json
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.testing import run_opt_pass
from tvm.relay.testing import Prelude
from tvm.ir import IRModule
from tvm import topi
from tvm.contrib import graph_runtime
from tvm.contrib.nvcc import have_fp16
from tvm.relay import Any
from tvm.relay.ty import TypeRelation
from tvm.relay.ty import TypeCall
import tvm.relay.testing
from tvm.relay import transform
from tvm.relay import create_executor
from tvm.relay.analysis import Feature
from tvm.runtime import container
from tvm.relay.transform import SimplifyInference
from tvm.relay import TypeMutator
import itertools
from tvm.relay import TypeFunctor
from tvm import runtime
from tvm.relay import ExprVisitor
import tvm.topi.testing
from tvm.relay.ty import TupleType
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.testing import check_grad
import tvm.relay.transform as _transform
from tvm import relay
import tvm.testing
from tvm.relay import testing
from tvm.relay import analysis
import tvm
from tvm.relay.testing import run_infer_type
from scipy import special
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.transform import to_cps
from tvm.relay.ty import TensorType
from tvm.relay.testing import count
from tvm.relay.transform import FastMath
from tvm.relay.prelude import Prelude
import math
from tvm.relay.backend.interpreter import ConstructorValue
from tvm import te
from tvm.relay.analysis import check_kind
from typing import Union
import scipy.sparse as sp
from tvm.relay import op
import pytest
from tvm.relay.op.annotation import compiler_begin
import tvm.relay.transform
from tvm.relay.testing import create_workload
from tvm.relay.analysis import well_formed
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.ty import RefType
from tvm.ir import structural_equal
from tvm.relay.ty import FuncType



def has_func_type(t):

    class FuncTypeVisitor(TypeVisitor):

        def __init__(self):
            super().__init__()
            self.has_func = False

        def visit_func_type(self, ftt):
            self.has_func = True
    ftvisitor = FuncTypeVisitor()
    ftvisitor.visit(t)
    return ftvisitor.has_func


def assert_no_higher_order_functions(expr, mod):

    class CheckFirstOrderVisitor(ExprVisitor):

        def __init__(self, mod):
            super().__init__()
            self.mod = mod
            self.hof = []
            self.visited_gv = set()

        def visit_call(self, call):
            is_higher_order = False
            if has_func_type(call.checked_type):
                is_higher_order = True
            for a in call.args:
                if has_func_type(a.checked_type):
                    is_higher_order = True
            if is_higher_order:
                self.hof.append(call)
            super().visit_call(call)

        def visit_global_var(self, gv):
            if (gv not in self.visited_gv):
                self.visited_gv.add(gv)
                self.visit(self.mod[gv])
    mod = transform.InferType()(mod)
    check_fo_visitor = CheckFirstOrderVisitor(mod)
    check_fo_visitor.visit(expr)
    nl = '\n--------\n'
    errmsg = f'''found {len(check_fo_visitor.hof)} higher order functions:
  {nl.join((expr.astext() for expr in check_fo_visitor.hof))}'''
    assert (len(check_fo_visitor.hof) == 0), errmsg


def defunctionalized(mod):
    mod = transform.InferType()(mod)
    mod['main'] = transform.Defunctionalization(mod['main'], mod)
    mod = transform.InferType()(mod)
    assert_no_higher_order_functions(mod['main'], mod)
    return mod


def to_list(mod, l):
    list = mod.get_global_type_var('List')
    list_adt = mod[list]
    cons = list_adt.constructors[0]
    nil = list_adt.constructors[1]
    assert isinstance(l, ConstructorValue)
    val = l
    ret = []
    while True:
        if (val.tag == cons.tag):
            ret.append(val.fields[0].asnumpy())
            val = val.fields[1]
        else:
            assert (val.tag == nil.tag)
            break
    return ret


def to_adt_list(mod, arr):
    expr = mod['main']
    l = mod.get_global_type_var('List')
    list_adt = mod[l]
    cons = list_adt.constructors[0]
    nil = list_adt.constructors[1]
    li = nil()
    for a in arr:
        li = cons(relay.const(a), li)
    ex = relay.create_executor(mod=mod)
    adt = ex.evaluate(li)
    mod['main'] = expr
    return adt


def has_func_type(t):

    class FuncTypeVisitor(TypeVisitor):

        def __init__(self):
            super().__init__()
            self.has_func = False

        def visit_func_type(self, ftt):
            self.has_func = True
    ftvisitor = FuncTypeVisitor()
    ftvisitor.visit(t)
    return ftvisitor.has_func


def assert_no_higher_order_functions(expr, mod):

    class CheckFirstOrderVisitor(ExprVisitor):

        def __init__(self, mod):
            super().__init__()
            self.mod = mod
            self.hof = []
            self.visited_gv = set()

        def visit_call(self, call):
            is_higher_order = False
            if has_func_type(call.checked_type):
                is_higher_order = True
            for a in call.args:
                if has_func_type(a.checked_type):
                    is_higher_order = True
            if is_higher_order:
                self.hof.append(call)
            super().visit_call(call)

        def visit_global_var(self, gv):
            if (gv not in self.visited_gv):
                self.visited_gv.add(gv)
                self.visit(self.mod[gv])
    mod = transform.InferType()(mod)
    check_fo_visitor = CheckFirstOrderVisitor(mod)
    check_fo_visitor.visit(expr)
    nl = '\n--------\n'
    errmsg = f'''found {len(check_fo_visitor.hof)} higher order functions:
  {nl.join((expr.astext() for expr in check_fo_visitor.hof))}'''
    assert (len(check_fo_visitor.hof) == 0), errmsg


def defunctionalized(mod):
    mod = transform.InferType()(mod)
    mod['main'] = transform.Defunctionalization(mod['main'], mod)
    mod = transform.InferType()(mod)
    assert_no_higher_order_functions(mod['main'], mod)
    return mod
purPt=tvm.parser.fromtext('''
#[version = "0.0.5"]
type List[A] {
  Cons(A, List[A]),
  Nil,
}
def @id[A](%x: A) -> A {
  %x
}
def @map[A, B](%f: fn(A) -> B, %xs: List[A]) -> List[B] {
  match (%xs) {
    Cons(%x, %rest) => Cons(%f(%x), @map(%f, %rest)),
    Nil => Nil,
  }
}
def @main(%l: List[float32]) -> List[float32] {
  @map(@id, %l)
}
''')
MCgLE=defunctionalized(purPt)
L4Nbl=np.random.rand(5)
L6DD3=L4Nbl.astype('''float64''')
DjwYV=to_adt_list(MCgLE,L6DD3)
QdQ1M=transform.LazyGradientInit()
TLzMc=tvm.IRModule()


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
SFg6q=relay.scalar_type('''int32''')
nfwcW=relay.FuncType([],SFg6q)
tMGKG=relay.Var('''f''',nfwcW)
MQEnD=tMGKG()
gYf0p=relay.const(0)
MNCnQ=relay.If(MQEnD,gYf0p,gYf0p)
Bon7C=relay.Function([tMGKG,tMGKG,],MNCnQ)
Y3FWM=tvm.IRModule({})
Y3FWM['''g1''']=Bon7C
Y3FWM['''g0''']=Bon7C
Y3FWM['''g1''']=Bon7C
TLzMc['''main''']=Bon7C
TLzMc['''main''']=Bon7C
rBkDC=run_infer_type(Bon7C)
TLzMc['''main''']=rBkDC
b7ytO=QdQ1M(TLzMc)
RdNHy=create_executor(mod=b7ytO)
TLzMc['''main''']=rBkDC
MH7Xb=RdNHy.evaluate(TLzMc['''main'''])
Fl61h=MH7Xb(DjwYV)


def has_func_type(t):

    class FuncTypeVisitor(TypeVisitor):

        def __init__(self):
            super().__init__()
            self.has_func = False

        def visit_func_type(self, ftt):
            self.has_func = True
    ftvisitor = FuncTypeVisitor()
    ftvisitor.visit(t)
    return ftvisitor.has_func


def assert_no_higher_order_functions(expr, mod):

    class CheckFirstOrderVisitor(ExprVisitor):

        def __init__(self, mod):
            super().__init__()
            self.mod = mod
            self.hof = []
            self.visited_gv = set()

        def visit_call(self, call):
            is_higher_order = False
            if has_func_type(call.checked_type):
                is_higher_order = True
            for a in call.args:
                if has_func_type(a.checked_type):
                    is_higher_order = True
            if is_higher_order:
                self.hof.append(call)
            super().visit_call(call)

        def visit_global_var(self, gv):
            if (gv not in self.visited_gv):
                self.visited_gv.add(gv)
                self.visit(self.mod[gv])
    mod = transform.InferType()(mod)
    check_fo_visitor = CheckFirstOrderVisitor(mod)
    check_fo_visitor.visit(expr)
    nl = '\n--------\n'
    errmsg = f'''found {len(check_fo_visitor.hof)} higher order functions:
  {nl.join((expr.astext() for expr in check_fo_visitor.hof))}'''
    assert (len(check_fo_visitor.hof) == 0), errmsg


def defunctionalized(mod):
    mod = transform.InferType()(mod)
    mod['main'] = transform.Defunctionalization(mod['main'], mod)
    mod = transform.InferType()(mod)
    assert_no_higher_order_functions(mod['main'], mod)
    return mod


def to_list(mod, l):
    list = mod.get_global_type_var('List')
    list_adt = mod[list]
    cons = list_adt.constructors[0]
    nil = list_adt.constructors[1]
    assert isinstance(l, ConstructorValue)
    val = l
    ret = []
    while True:
        if (val.tag == cons.tag):
            ret.append(val.fields[0].asnumpy())
            val = val.fields[1]
        else:
            assert (val.tag == nil.tag)
            break
    return ret
NiGSd=relay.create_executor('''debug''',mod=MCgLE)
wCBSw=NiGSd.evaluate()
SqxAA=wCBSw(DjwYV)
txe9C=to_list(purPt,SqxAA)
np.testing.assert_array_equal(txe9C,txe9C)
