from typing import Union
from tvm.relay.ty import RefType
from tvm import topi
from tvm.relay.transform import un_cps
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.testing import Prelude
from tvm.relay.analysis import well_formed
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.testing import make_nat_expr
import itertools
from tvm.relay import TypeMutator
import scipy
import pytest
from tvm.relay.transform import to_cps
from tvm.relay.testing import rand
from tvm.testing import assert_allclose
from tvm.relay.analysis import Feature
from tvm.contrib.nvcc import have_fp16
from tvm import relay
import random
from tvm.contrib import graph_runtime
from tvm.relay.testing import check_grad
from tvm.relay.analysis import get_calibration_data
from tvm.relay import op
import tvm.relay.transform as _transform
import math
from tvm.relay import ExprVisitor
from tvm.relay import TypeVisitor
from functools import wraps
import json
from numpy import isclose
import sys
from tvm.autotvm.tuner import RandomTuner
from scipy import special
from tvm.relay.analysis import check_kind
import tvm.topi.testing
from tvm.relay.ty import FuncType
from tvm.relay.ty import TypeRelation
from tvm import relay as rly
import tvm.relay.testing
from tvm.relay import expr as _expr
from tvm.relay import Any
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay.testing import enabled_targets
from tvm.relay.testing import run_opt_pass
from tvm.relay.testing import run_infer_type
from tvm.relay.testing.synthetic import get_workload
import tvm.relay as relay
import logging
from tvm.relay import transform
from tvm.runtime import container
from tvm import autotvm
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.ty import TensorType
from tvm.relay.ty import TypeCall
import os
import tvm
from tvm.relay.testing import count
from tvm.relay.transform import FastMath
import tvm.testing
from tvm.relay import TypeFunctor
from tvm.relay import create_executor
from tvm.relay.prelude import Prelude
from tvm.relay.ty import TypeVar
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.adt import TypeData
from tvm.ir import IRModule
import numpy as np
from tvm.relay.transform import SimplifyInference
from tvm import te
from tvm import nd
from tvm.relay import analysis
from tvm.ir import structural_equal
from tvm.relay.analysis import detect_feature
from tvm.relay.ty import TupleType
from tvm import runtime
import time
from tvm.relay.testing import create_workload
from tvm.relay import testing
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.backend.interpreter import ConstructorValue
import scipy.sparse as sp
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.backend.interpreter import RefValue
import tvm.relay.transform
from tvm.relay.op.annotation import compiler_end
from tvm.relay.ty import IncompleteType

Kujmb=tvm.gpu()
mipga=relay.IncompleteType()
UmwKv=relay.Var('''a''',mipga)
ePKK2=relay.op.memory.alloc_storage(UmwKv,UmwKv,Kujmb)
