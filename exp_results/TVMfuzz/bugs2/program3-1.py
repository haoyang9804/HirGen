import tvm.relay.transform
from tvm.relay.testing import rand
from tvm.contrib import graph_runtime
from tvm.relay.transform import to_cps
from tvm.relay import testing
from tvm.relay import expr as _expr
from tvm.relay import analysis
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.transform import SimplifyInference
import random
from tvm.relay.testing import check_grad
import time
from tvm.relay import create_executor
import logging
import math
import numpy as np
from tvm.relay.op.annotation import compiler_end
from tvm.ir import IRModule
from tvm.relay.ty import TensorType
from tvm import relay
from tvm.relay.ty import TupleType
from tvm.relay.analysis import check_basic_block_normal_form
from numpy import isclose
from functools import wraps
from tvm.relay import TypeMutator
from tvm.runtime import container
from tvm.relay.adt import TypeData
from tvm.relay.analysis import Feature
from tvm.relay import transform
import pytest
from tvm.relay.analysis import get_calibration_data
from tvm import nd
from tvm.relay.ty import FuncType
from tvm.relay.testing import enabled_targets
import tvm
import scipy.sparse as sp
from tvm.relay.ty import TypeCall
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.testing import create_workload
import tvm.relay.transform as _transform
from tvm.ir import structural_equal
from tvm.relay.testing import run_infer_type
import itertools
from tvm import autotvm
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.testing.synthetic import get_workload
from tvm.relay import ExprVisitor
from tvm.relay import TypeFunctor
import scipy
from tvm import te
import sys
from tvm import topi
from tvm.relay.testing import count
import json
import os
from tvm.relay.analysis import detect_feature
from tvm.relay.transform import FastMath
from scipy import special
from tvm.relay.testing import make_nat_expr
from tvm.relay import Any
import tvm.topi.testing
from tvm.relay.ty import TypeVar
from tvm.relay.analysis import well_formed
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm import relay as rly
from tvm.contrib.nvcc import have_fp16
from tvm.relay.analysis import check_kind
from typing import Union
from tvm.relay.ty import TypeRelation
from tvm.testing import assert_allclose
import tvm.testing
from tvm.relay.testing import Prelude
from tvm.relay import op
from tvm.autotvm.tuner import RandomTuner
from tvm import runtime
from tvm.relay.backend.interpreter import ConstructorValue
import tvm.relay.testing
from tvm.relay.ty import RefType
from tvm.relay.ty import IncompleteType
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.prelude import Prelude
from tvm.relay import TypeVisitor
from tvm.relay.transform import un_cps
from tvm.relay.build_module import bind_params_by_name
import tvm.relay as relay
from tvm.relay.testing import run_opt_pass
from tvm.relay.scope_builder import ScopeBuilder



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
PP9An=tvm.parser.fromtext('''
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
RDGbL=defunctionalized(PP9An)
zW7zH=relay.create_executor('''debug''',mod=RDGbL)
Ze1pm=zW7zH.evaluate()


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
z0QyD=np.random.rand(10)
YP4DL=z0QyD.astype('''float64''')
jf0sc=to_adt_list(PP9An,YP4DL)
pNWhO=Ze1pm(jf0sc)


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
sQSCp=relay.transform.ToBasicBlockNormalForm()
VgxAh=relay.scalar_type('''float64''')
aDw1E=relay.Var('''size''',VgxAh)
T8rXj=relay.var('''cond''',shape=(),dtype='''float64''')
HFWxS=relay.Var('''uv''')
SjcQN=relay.const(0)
Xnz8E=relay.RefWrite(HFWxS,SjcQN)
yGzXn=relay.const(0.0,dtype='''float64''')
NCgJ6=relay.If(T8rXj,Xnz8E,yGzXn)
u2KWZ=relay.var('''data1''',shape=(1,16,))
iiPLk=relay.nn.conv2d(u2KWZ,u2KWZ,channels=64,kernel_size=(3,3,),padding=(1,1,))
zBKuS=relay.add(NCgJ6,iiPLk)
Q3Yrh=relay.Function([aDw1E,aDw1E,],zBKuS)
hyCU3=tvm.IRModule.from_expr(Q3Yrh)
GZRs3=sQSCp(hyCU3)
HE31R=tvm.IRModule()
HE31R['''main''']=Q3Yrh
En0b3=relay.transform.gradient(HE31R['''main'''],mode='''higher_order''')
GZRs3['''main''']=En0b3
XFtgN=zW7zH.evaluate(GZRs3['''main'''])
w6GOQ=XFtgN(jf0sc)
f510U=to_list(PP9An,w6GOQ)
np.testing.assert_array_equal(f510U,f510U)
