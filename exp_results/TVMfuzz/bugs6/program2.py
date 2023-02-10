from tvm.relay import TypeVisitor
from tvm.relay.ty import TypeVar
import tvm.topi.testing
from tvm.relay.op.annotation import compiler_begin
import tvm.relay.transform
from tvm.runtime import container
from tvm.relay.testing.synthetic import get_workload
from tvm.relay import TypeFunctor
from tvm.relay.transform import to_cps
from tvm.relay.testing import rand
from tvm.relay import op
from tvm.relay import expr as _expr
from tvm.relay.ty import TypeRelation
from tvm.relay import testing
from tvm.relay import ExprVisitor
from numpy import isclose
from tvm import topi
import scipy
from tvm import relay
import math
from tvm.relay.ty import TypeCall
from tvm.relay.testing import make_nat_expr
from tvm.relay import Any
from tvm import runtime
import time
from tvm.relay.testing import enabled_targets
import tvm.testing
from tvm.relay.ty import TensorType
import numpy as np
from tvm.relay import create_executor
from tvm.testing import assert_allclose
from tvm.relay.analysis import well_formed
from tvm.ir import IRModule
from tvm.relay.testing import run_opt_pass
from tvm.relay.testing import check_grad
from tvm.relay.analysis import get_calibration_data
from tvm.relay import analysis
from tvm.relay.testing import Prelude
from tvm.relay.ty import FuncType
import tvm
from tvm.relay.analysis import Feature
from tvm.relay.transform import FastMath
import random
import sys
from tvm.relay.backend.interpreter import ConstructorValue
import pytest
from tvm.relay.ty import RefType
from tvm.relay import transform
from tvm.relay.ty import GlobalTypeVar
import os
from tvm.relay.transform import SimplifyInference
import tvm.relay.transform as _transform
from scipy import special
import tvm.relay.testing
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.annotation import compiler_end
from tvm import nd
from typing import Union
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.adt import TypeData
from tvm.relay.prelude import Prelude
from tvm.relay.ty import TupleType
from tvm import te
from functools import wraps
from tvm.relay.ty import IncompleteType
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.analysis import check_kind
import scipy.sparse as sp
import tvm.relay as relay
from tvm.relay.build_module import bind_params_by_name
import itertools
from tvm.relay.testing import create_workload
from tvm.relay import TypeMutator
from tvm.ir import structural_equal
import json
from tvm.contrib.nvcc import have_fp16
from tvm.relay.testing import run_infer_type
from tvm.relay.testing import count
from tvm.autotvm.tuner import RandomTuner
from tvm import relay as rly
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.contrib import graph_runtime
from tvm.relay.transform import un_cps
from tvm import autotvm
import logging
from tvm.relay.analysis import detect_feature

pfy4X=tvm.gpu()
OLuoO=relay.scalar_type('''uint64''')
L0BT9=relay.Var('''y''',OLuoO)
a5Txo=relay.op.memory.alloc_storage(L0BT9,L0BT9,pfy4X)
