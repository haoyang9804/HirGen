from tvm.relay.testing import create_workload
from tvm.relay.testing import run_opt_pass
from tvm.relay.ty import RefType
import os
from tvm.relay import analysis
import scipy
from tvm.relay.scope_builder import ScopeBuilder
from tvm import relay
from tvm.relay.transform import FastMath
from tvm.relay import TypeMutator
from tvm.relay import create_executor
from tvm.relay.ty import TypeCall
import scipy.sparse as sp
from tvm.relay import TypeFunctor
from tvm.relay.analysis import get_calibration_data
from tvm.relay.analysis import detect_feature
import tvm.relay.transform as _transform
from tvm.relay.prelude import Prelude
from tvm import relay as rly
from tvm.ir import structural_equal
from tvm.relay import expr as _expr
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.ty import TupleType
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.testing import Prelude
from tvm.relay.transform import un_cps
import pytest
import tvm.relay.transform
from tvm.relay.ty import IncompleteType
import numpy as np
from tvm.relay.analysis import check_kind
from tvm.relay import ExprVisitor
from tvm.relay.transform import to_cps
from tvm.relay.adt import TypeData
from tvm.relay import Any
from tvm.relay.ty import FuncType
from tvm.relay.analysis import well_formed
import logging
from tvm import autotvm
from tvm.testing import assert_allclose
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import testing
from tvm.relay import transform
import tvm.testing
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.ty import TypeRelation
from tvm.relay import op
from tvm.relay.ty import TensorType
from tvm.relay.testing import run_infer_type
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay import TypeVisitor
from tvm.runtime import container
from tvm.relay.backend.interpreter import ConstructorValue
import sys
from tvm.ir import IRModule
import math
from tvm import nd
import random
from tvm import topi
from tvm.relay.testing import check_grad
from tvm.relay.testing import count
from tvm.relay.op.annotation import compiler_begin
import json
import time
from tvm import runtime
from tvm.relay.op.annotation import compiler_end
from tvm import te
from tvm.relay.analysis import Feature
from tvm.contrib.nvcc import have_fp16
from tvm.relay.testing import rand
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.ty import TypeVar
from tvm.contrib import graph_runtime
from functools import wraps
import tvm
from tvm.relay.testing import enabled_targets
from tvm.relay.transform import SimplifyInference
from tvm.relay.testing.temp_op_attr import TempOpAttr
import tvm.relay as relay
from tvm.autotvm.tuner import RandomTuner
import tvm.topi.testing
from tvm.relay.testing import make_nat_expr
import itertools
from numpy import isclose
from typing import Union
from scipy import special
import tvm.relay.testing

OSabL=relay.scalar_type('''int64''')
dDXaJ=relay.Var('''f''',OSabL)
a1x3A=dDXaJ()
vpFwM=relay.FuncType([],OSabL)
z3CKd=relay.TypeVar('''a''')
vsC6H=relay.Function([dDXaJ,],dDXaJ,vpFwM,[z3CKd,])
zvNDj=relay.Var('''y''')
i8W09=tvm.gpu()
AOQhI=relay.op.memory.alloc_storage(dDXaJ,dDXaJ,i8W09)
Hwhvn=relay.Function([zvNDj,],AOQhI)
Ns6ml=relay.If(a1x3A,vsC6H,Hwhvn)
