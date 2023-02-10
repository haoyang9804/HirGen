from functools import wraps
from tvm.relay.ty import TypeCall
from tvm.relay.ty import TensorType
from tvm.relay.testing import Prelude
import tvm.relay.transform
from tvm.relay.transform import un_cps
from tvm.autotvm.tuner import RandomTuner
import time
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.testing import enabled_targets
import json
from tvm.relay.analysis import detect_feature
from tvm.relay.ty import IncompleteType
from tvm.relay.ty import TupleType
from tvm.relay.ty import GlobalTypeVar
from tvm import relay
from tvm.relay.testing import check_grad
import os
from tvm.relay.ty import FuncType
from tvm import autotvm
import tvm.relay as relay
import numpy as np
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.testing import rand
from tvm.relay.analysis import check_basic_block_normal_form
import tvm.relay.testing
from tvm import nd
from tvm.relay import expr as _expr
from tvm.relay.testing import make_nat_expr
from tvm.relay.analysis import check_kind
import itertools
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.analysis import get_calibration_data
import random
import scipy.sparse as sp
from tvm.relay.adt import TypeData
import tvm.relay.transform as _transform
from tvm.relay.testing import run_opt_pass
from tvm.runtime import container
import logging
import tvm.topi.testing
from tvm.relay import TypeMutator
from tvm.relay.testing import count
from tvm.relay.scope_builder import ScopeBuilder
import tvm.testing
from tvm.relay.analysis import Feature
from tvm.relay.ty import TypeRelation
from tvm.relay import analysis
from tvm.relay.testing import create_workload
from tvm.relay.testing import run_infer_type
from scipy import special
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay import testing
import math
from typing import Union
from tvm.relay import ExprVisitor
from tvm.ir import structural_equal
from tvm import te
from tvm.relay.analysis import well_formed
from tvm.relay.prelude import Prelude
from tvm.relay import transform
import tvm
from tvm import relay as rly
from numpy import isclose
import sys
from tvm.relay.ty import RefType
from tvm import topi
from tvm.testing import assert_allclose
from tvm.relay.transform import FastMath
import pytest
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.ir import IRModule
from tvm.relay import Any
from tvm.relay.op.annotation import compiler_begin
import scipy
from tvm.relay import TypeFunctor
from tvm.relay.transform import SimplifyInference
from tvm.relay.ty import TypeVar
from tvm.relay.op.annotation import compiler_end
from tvm.relay import TypeVisitor
from tvm.relay import op
from tvm.relay import create_executor
from tvm.contrib import graph_runtime
from tvm.relay.transform import to_cps
from tvm.contrib.nvcc import have_fp16
from tvm import runtime

Hu0mW=tvm.gpu()
XPeuN=relay.scalar_type('''int64''')
l97HZ=relay.Var('''x''',XPeuN)
FCVjg=relay.op.memory.alloc_storage(l97HZ,l97HZ,Hu0mW)
