from compile.compile_err import CompilationError

class XlaError(CompilationError):
    def get_err_code(self):
        return "None"