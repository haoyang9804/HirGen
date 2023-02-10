import tvm.topi.testing
from tvm.relay.backend.interpreter import RefValue
from scipy import special
from tvm.ir import IRModule
from tvm import relay
from typing import Union
from tvm.relay.ty import TensorType
import numpy as np
from tvm.runtime import container
from tvm.relay.testing import rand
import logging
import tvm.relay.testing
from tvm.testing import assert_allclose
from functools import wraps
from tvm.relay import Any
from tvm.relay.prelude import Prelude
from tvm.relay.ty import IncompleteType
import scipy.sparse as sp
from tvm.relay.ty import TypeCall
import os
import time
from tvm.relay.testing import Prelude
import random
from tvm.relay.ty import RefType
from tvm.contrib.nvcc import have_fp16
from tvm.relay.scope_builder import ScopeBuilder
import tvm
from tvm.relay.testing import enabled_targets
import tvm.relay as relay
from tvm.relay.transform import SimplifyInference
from tvm.relay.transform import FastMath
import math
from tvm.relay.testing import make_nat_expr
from numpy import isclose
from tvm.relay import analysis
from tvm import nd
from tvm.relay import op
import sys
from tvm.relay import create_executor
import tvm.relay.transform
from tvm.relay.transform import un_cps
from tvm.relay import TypeVisitor
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.annotation import compiler_end
from tvm.autotvm.tuner import RandomTuner
from tvm.relay.adt import TypeData
from tvm.relay.data_dep_optimization import simplify_fc_transpose
import scipy
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.ty import FuncType
from tvm.relay.testing import create_workload
from tvm.relay.analysis import get_calibration_data
from tvm.relay.analysis import detect_feature
from tvm import te
from tvm.relay import expr as _expr
from tvm.relay import TypeMutator
from tvm.relay.testing import count
from tvm.relay.analysis import Feature
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.ty import TypeRelation
from tvm import runtime
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.ty import TupleType
import pytest
from tvm.relay import testing
import tvm.relay.transform as _transform
from tvm.relay.ty import GlobalTypeVar
from tvm import autotvm
from tvm.ir import structural_equal
from tvm.contrib import graph_runtime
from tvm.relay.testing.synthetic import get_workload
import tvm.testing
from tvm.relay.testing import run_infer_type
from tvm.relay.testing import check_grad
import itertools
from tvm.relay.analysis import well_formed
import json
from tvm.relay import TypeFunctor
from tvm.relay import ExprVisitor
from tvm.relay.testing import run_opt_pass
from tvm.relay.transform import to_cps
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay.ty import TypeVar
from tvm.relay.analysis import check_kind
from tvm import topi
from tvm import relay as rly
from tvm.relay import transform

UtC3w=transform.LazyGradientInit()
N6dGf=tvm.IRModule()
qvL4c=relay.scalar_type('''int16''')
v54qV=relay.var('''x''',qvL4c)
ol2bK=relay.Function([v54qV,v54qV,],(v54qV + v54qV))
N6dGf['''main''']=ol2bK
x8jC0=relay.GlobalTypeVar('''Either''')
g2YJr=relay.TypeVar('''b''')
yhk3q=relay.Constructor('''Right''',[g2YJr,],x8jC0)
maRoz=relay.TypeData(x8jC0,[g2YJr,g2YJr,],[yhk3q,yhk3q,])
N6dGf[x8jC0]=maRoz
vVeNj=UtC3w(N6dGf)
