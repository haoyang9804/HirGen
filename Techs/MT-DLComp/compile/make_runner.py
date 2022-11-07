from utils.path_utils import norm_user_path

def make_runner(compiler_name, compiler_path, mode, cal_time):
    if compiler_path:
        compiler_path = norm_user_path(compiler_path)
    if compiler_name == 'glow':
        from compile.glow.glow import GlowRunner
        return GlowRunner(compiler_path, mode, cal_time)
    elif compiler_name == 'tvm':
        from compile.tvm.tvm import TVMRunner
        return TVMRunner(compiler_path, mode, cal_time)
    elif compiler_name == 'xla':
        from compile.xla.xla import XlaRunner
        return XlaRunner(compiler_path, mode, cal_time)
    elif compiler_name == 'tensorflow':
        from compile.tf_runner.tf_runner import TfRunner
        return TfRunner(compiler_path, mode, cal_time)
    elif compiler_name == 'onnx':
        from compile.onnx_runner.onnx_runner import OnnxRunner
        return OnnxRunner(compiler_path, mode, cal_time)
    elif compiler_name == 'glow-zoo':
        from cross_compile.glow_compile import GlowModelZooCompiler
        return GlowModelZooCompiler(compiler_path)
    else:
        return None