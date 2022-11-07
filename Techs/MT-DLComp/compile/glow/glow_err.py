import re

from compile.compile_err import CompilationError

class GlowError(CompilationError):
    def __init__(self, model_path, err_info):
        super(GlowError, self).__init__(model_path, err_info)

    def get_err_code(self):
        m = re.search("Error code: (\w+)", self.err_info)
        if m:
            return m.group(1)
        else:
            return None
