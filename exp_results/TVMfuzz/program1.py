import tvm.relay.transform
from tvm.relay import transform
import numpy as np
from tvm.relay.transform import un_cps
import tvm.topi.testing
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay.transform import FastMath
from tvm.relay.op.annotation import compiler_end
from tvm.relay.testing import make_nat_expr
from tvm.relay.testing import Prelude
import time
from scipy import special
from tvm.relay.testing import rand
from tvm.relay.data_dep_optimization import simplify_fc_transpose
import scipy.sparse as sp
from tvm.relay.ty import TensorType
import tvm.relay as relay
from tvm.relay.analysis import get_calibration_data
import os
from tvm import runtime
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.ty import RefType
from tvm.relay.build_module import bind_params_by_name
import pytest
from tvm.autotvm.tuner import RandomTuner
from tvm.relay.testing import run_infer_type
import scipy
import logging
from tvm.relay import TypeVisitor
from tvm.relay.adt import TypeData
from tvm import autotvm
import sys
from typing import Union
import math
from tvm.relay.ty import FuncType
from tvm.relay.ty import TypeRelation
from tvm.runtime import container
from tvm import relay as rly
from numpy import isclose
import random
from functools import wraps
from tvm.ir import structural_equal
import tvm
from tvm.testing import assert_allclose
from tvm.contrib.nvcc import have_fp16
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.prelude import Prelude
from tvm.relay import expr as _expr
from tvm import nd
from tvm.relay.ty import IncompleteType
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.ty import TypeCall
from tvm.contrib import graph_runtime
from tvm.relay import testing
from tvm.relay.testing import count
from tvm import relay
from tvm.relay import op
import tvm.testing
from tvm.relay import TypeFunctor
from tvm.relay.testing import check_grad
from tvm.relay.testing import create_workload
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.ty import TypeVar
import tvm.relay.transform as _transform
import tvm.relay.testing
from tvm.relay.analysis import check_kind
from tvm.relay import create_executor
from tvm.relay.testing import run_opt_pass
from tvm import topi
from tvm.relay.ty import TupleType
from tvm.relay.analysis import detect_feature
from tvm.relay import ExprVisitor
import json
from tvm.relay.analysis import well_formed
from tvm.relay.testing.synthetic import get_workload
from tvm.ir import IRModule
from tvm.relay import analysis
from tvm.relay import TypeMutator
from tvm.relay.analysis import Feature
from tvm.relay.transform import SimplifyInference
from tvm import te
from tvm.relay.transform import to_cps
import itertools
from tvm.relay.op.annotation import compiler_begin
from tvm.relay import Any
from tvm.relay.testing import enabled_targets

v8Ob9=relay.TypeVar('''a''')
jVnfi=relay.GlobalTypeVar('''id''')
GQd2d=jVnfi(v8Ob9)
cRKhe=relay.FuncType([v8Ob9,],GQd2d,[v8Ob9,])
g7wdx=relay.scalar_type('''float32''')
G6oRq=relay.Var('''make_id''',g7wdx)
HGe1O=tvm.gpu()
YmEe7=relay.op.memory.alloc_storage(G6oRq,G6oRq,HGe1O)
Kgh6s=G6oRq(G6oRq)
