from functools import wraps
from tvm.relay.ty import TupleType
from tvm import topi
from tvm.ir import structural_equal
from tvm.relay.ty import FuncType
from tvm.relay.testing import count
from tvm.relay.analysis import detect_feature
from tvm.testing import assert_allclose
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.transform import un_cps
from tvm.relay import testing
from tvm.ir import IRModule
from tvm.relay import TypeFunctor
import tvm
from tvm.relay.ty import TypeVar
from tvm.relay.transform import FastMath
from tvm.relay import op
from tvm.relay import create_executor
from tvm import relay as rly
import sys
from scipy import special
from tvm.relay.backend.interpreter import RefValue
from tvm import relay
from tvm.relay.analysis import well_formed
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay import Any
from tvm.relay.ty import TensorType
from tvm.relay.transform import SimplifyInference
from tvm.relay.ty import RefType
import tvm.relay.transform
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.testing import run_infer_type
from tvm.relay.ty import IncompleteType
from tvm.relay.testing import rand
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.prelude import Prelude
from tvm.relay.testing import check_grad
from tvm.runtime import container
from typing import Union
from tvm.relay.ty import GlobalTypeVar
from numpy import isclose
import numpy as np
from tvm.relay.adt import TypeData
import json
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay.analysis import Feature
from tvm.relay import transform
import scipy.sparse as sp
from tvm.relay.ty import TypeRelation
import tvm.relay.transform as _transform
from tvm import nd
from tvm.relay.ty import TypeCall
from tvm.relay.testing import create_workload
from tvm.relay import TypeMutator
import math
import tvm.relay as relay
import scipy
from tvm.relay.testing import Prelude
from tvm.relay import analysis
from tvm import autotvm
from tvm.relay.op.annotation import compiler_end
from tvm.relay.analysis import get_calibration_data
import time
from tvm.relay.backend.interpreter import ConstructorValue
import tvm.testing
from tvm import te
import tvm.relay.testing
from tvm.autotvm.tuner import RandomTuner
import random
from tvm.relay.analysis import check_kind
from tvm.contrib import graph_runtime
from tvm.relay.testing import enabled_targets
from tvm.relay import expr as _expr
from tvm.contrib.nvcc import have_fp16
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import TypeVisitor
from tvm.relay import ExprVisitor
from tvm import runtime
from tvm.relay.transform import to_cps
import itertools
from tvm.relay.testing import run_opt_pass
from tvm.relay.data_dep_optimization import simplify_fc_transpose
import os
import pytest
from tvm.relay.testing import make_nat_expr
import tvm.topi.testing
import logging

VClcQ=tvm.gpu()
WFdN7=relay.TypeVar('''xt''')
TSZwe=relay.Var('''alignment''',WFdN7)
EU9JA=relay.op.memory.alloc_storage(TSZwe,TSZwe,VClcQ)
