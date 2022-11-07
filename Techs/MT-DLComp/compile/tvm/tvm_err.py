from compile.compile_err import CompilationError

class TvmError(CompilationError):
    def get_err_code(self):
        return "None"