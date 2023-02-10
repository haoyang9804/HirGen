from tvm.contrib.nvcc import have_fp16
import itertools
import sys
from tvm.relay import Any
from tvm.relay.analysis import check_kind
from tvm.relay.adt import TypeData
from tvm import relay as rly
from tvm import relay
import tvm.relay.transform as _transform
import tvm.relay.testing
from tvm.relay import ExprVisitor
from tvm.relay.testing import Prelude
from tvm.relay.op.annotation import compiler_begin
from scipy import special
from tvm.relay import testing
from tvm.autotvm.tuner import RandomTuner
from tvm.relay.ty import RefType
from tvm.relay.ty import FuncType
from tvm.relay.analysis import get_calibration_data
import tvm
from tvm.ir import structural_equal
from tvm.relay.transform import SimplifyInference
from tvm.relay.ty import TypeCall
import tvm.relay as relay
from numpy import isclose
from tvm.relay.analysis import Feature
from tvm.relay import op
from tvm.relay.ty import TupleType
from tvm.relay.ty import TensorType
from tvm.relay import TypeMutator
from tvm.relay.scope_builder import ScopeBuilder
from tvm import nd
from tvm.ir import IRModule
from tvm.relay.prelude import Prelude
from tvm.relay.ty import IncompleteType
import scipy
from tvm.contrib import graph_runtime
from tvm.runtime import container
from tvm.relay import transform
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.analysis import detect_feature
from tvm.relay.testing import create_workload
from tvm.relay.testing import make_nat_expr
from tvm.relay.testing import rand
from tvm.relay.transform import un_cps
import scipy.sparse as sp
from tvm.relay.ty import TypeVar
from tvm.relay.testing.temp_op_attr import TempOpAttr
import tvm.relay.transform
import tvm.topi.testing
import os
from tvm import runtime
from tvm.relay.build_module import bind_params_by_name
import time
from tvm.relay.testing.synthetic import get_workload
import pytest
import json
from tvm.relay.testing import enabled_targets
from tvm.testing import assert_allclose
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay.testing import run_opt_pass
from tvm.relay import analysis
from typing import Union
from tvm import topi
from tvm.relay.ty import TypeRelation
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.testing import check_grad
from tvm.relay.testing import run_infer_type
from tvm.relay.op.annotation import compiler_end
from tvm.relay.transform import to_cps
import logging
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.testing import count
from tvm import autotvm
from tvm.relay.analysis import well_formed
from tvm.relay.transform import FastMath
import numpy as np
import random
import math
import tvm.testing
from tvm import te
from tvm.relay import TypeVisitor
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay import TypeFunctor
from functools import wraps
from tvm.relay import expr as _expr
from tvm.relay import create_executor

CTk2u=rand('''float64''',*(10,10,))
zlAUx=CTk2u.asnumpy()


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
ReWuf=tvm.parser.fromtext('''
#[version = "0.0.5"]
def @simple[A, B](%f: fn(A) -> B, %xs: A) -> B {
  %f(%xs)
}
def @main(%l: Tensor[(5, 5), float32]) -> Tensor[(5, 5), float32] {
  %0 = fn[A](%x: A) -> A {
    %x
  };
  @simple(%0, %l)
}
''')
INsgJ=defunctionalized(ReWuf)
GDJg7=relay.create_executor('''debug''',mod=INsgJ)
bbIoV=GDJg7.evaluate()
DVaSd=bbIoV(CTk2u)
JrAIQ=DVaSd.asnumpy()
tvm.testing.assert_allclose(JrAIQ,(0 * zlAUx))
