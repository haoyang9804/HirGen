import tvm.topi.testing
from tvm.relay.analysis import detect_feature
from tvm.relay import TypeVisitor
from tvm import nd
from tvm.relay.ty import TupleType
from tvm.relay import TypeFunctor
from tvm.testing import assert_allclose
import sys
import tvm.relay.testing
from tvm import relay as rly
from tvm.relay import create_executor
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.testing import check_grad
from tvm.relay.testing import make_nat_expr
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.ty import TypeVar
import tvm.relay.transform as _transform
import scipy.sparse as sp
from tvm.contrib.nvcc import have_fp16
from tvm import runtime
from tvm.relay.ty import TypeCall
import numpy as np
import tvm.relay.transform
from tvm.relay import expr as _expr
from tvm.ir import structural_equal
from tvm.relay.testing import count
from tvm.autotvm.tuner import RandomTuner
from tvm.relay.testing.synthetic import get_workload
from tvm.relay import testing
from tvm.relay import analysis
from tvm.relay.testing import Prelude
from tvm.relay.ty import RefType
from tvm.relay import transform
import os
from tvm.relay.transform import SimplifyInference
from tvm.relay.testing.temp_op_attr import TempOpAttr
import tvm.relay as relay
from tvm import autotvm
from tvm.relay.testing import rand
from tvm.relay.backend.interpreter import RefValue
from tvm import topi
from tvm.relay import ExprVisitor
from tvm.relay.analysis import get_calibration_data
import pytest
from typing import Union
import logging
from numpy import isclose
from tvm.relay.testing import run_opt_pass
from tvm import te
import tvm
from tvm.relay.transform import un_cps
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.testing import run_infer_type
from tvm.relay.transform import to_cps
from tvm.relay import op
from tvm import relay
from tvm.relay.adt import TypeData
import time
import json
from functools import wraps
from tvm.relay.op.annotation import compiler_end
from tvm.ir import IRModule
from tvm.runtime import container
from tvm.relay.ty import IncompleteType
from tvm.relay.analysis import check_kind
import tvm.testing
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay import TypeMutator
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.ty import TensorType
from scipy import special
from tvm.relay.testing import create_workload
import random
from tvm.relay.ty import TypeRelation
from tvm.relay.prelude import Prelude
from tvm.relay.ty import FuncType
import scipy
from tvm.relay.testing import enabled_targets
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.transform import FastMath
import itertools
from tvm.relay.analysis import Feature
from tvm.relay import Any
from tvm.contrib import graph_runtime
from tvm.relay.analysis import well_formed
import math

qz7mG=relay.scalar_type('''uint32''')
OypsU=relay.Var('''size''',qz7mG)
uH2bi=tvm.gpu()
fVDYG=relay.op.memory.alloc_storage(OypsU,OypsU,uH2bi)
