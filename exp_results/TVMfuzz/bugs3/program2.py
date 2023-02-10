from tvm.relay.op.annotation import compiler_begin
from tvm.relay.testing import run_opt_pass
import tvm.topi.testing
import tvm.relay.testing
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.ty import TensorType
from tvm.ir import structural_equal
from tvm.relay import TypeFunctor
import tvm.testing
from tvm.relay import Any
from tvm import nd
from tvm.relay.transform import un_cps
from tvm.relay.ty import TypeVar
import math
from tvm.ir import IRModule
from tvm.testing import assert_allclose
import itertools
from tvm.relay.backend.interpreter import ConstructorValue
import os
from tvm.relay.analysis import Feature
from tvm.relay.testing.synthetic import get_workload
import tvm.relay.transform as _transform
from tvm.autotvm.tuner import RandomTuner
from tvm.relay.testing import check_grad
import scipy.sparse as sp
from tvm.relay.testing import rand
from tvm.relay import testing
import scipy
import tvm.relay as relay
from tvm.relay import transform
import logging
from tvm.relay.transform import FastMath
from tvm.relay.ty import GlobalTypeVar
from tvm.relay import ExprVisitor
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.ty import RefType
from tvm.relay import op
import sys
from tvm import topi
from tvm.relay.op.annotation import compiler_end
from tvm.relay.adt import TypeData
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.ty import TupleType
from tvm.relay.testing import Prelude
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.analysis import check_basic_block_normal_form
from functools import wraps
from tvm import relay as rly
from tvm.relay.transform import to_cps
from typing import Union
from tvm import te
from tvm.relay.testing import make_nat_expr
from tvm.relay.scope_builder import ScopeBuilder
from tvm import runtime
from tvm.relay.ty import TypeRelation
from tvm import relay
import tvm.relay.transform
from tvm.relay.analysis import get_calibration_data
from tvm.relay.testing import run_infer_type
from numpy import isclose
from tvm.relay.ty import IncompleteType
from tvm.relay.testing import enabled_targets
from tvm.relay.testing import count
from tvm.relay.ty import FuncType
from tvm.relay import create_executor
import numpy as np
from tvm.relay import analysis
import pytest
from tvm.relay.testing import create_workload
from tvm.relay.analysis import detect_feature
from scipy import special
from tvm.contrib.nvcc import have_fp16
from tvm.relay import TypeMutator
import json
from tvm.relay.ty import TypeCall
from tvm.contrib import graph_runtime
from tvm.relay.analysis import check_kind
from tvm.runtime import container
from tvm.relay import expr as _expr
from tvm.relay.analysis import well_formed
import time
from tvm.relay.transform import SimplifyInference
import random
from tvm.relay.prelude import Prelude
from tvm.relay import TypeVisitor
import tvm
from tvm import autotvm

Fs42T=relay.scalar_type('''float64''')
J20OO=relay.Var('''size''',Fs42T)
S0gdu=tvm.gpu()
NK1Uc=relay.op.memory.alloc_storage(J20OO,J20OO,S0gdu)
