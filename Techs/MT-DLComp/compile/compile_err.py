class CompilationError(Exception):
    def __init__(self, model_path, err_info):
        self.model_path = model_path
        if isinstance(err_info, bytes):
            self.err_info = bytes.decode(err_info)
        else:
            self.err_info = err_info
        self.err_code = self.get_err_code()

    def get_err_code(self):
        raise NotImplementedError()

    def __str__(self):
        return "Compilation failed: %s $$$ %s $$$ %s" % (
            self.model_path, self.err_code, self.err_info)
