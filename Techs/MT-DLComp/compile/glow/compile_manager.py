import os
import shutil

from compile.glow.form_cpp import make_run_cpp, get_inout_info


class CompileManager:
    run_file = None

    @classmethod
    def form_run_cpp(cls, build_dir):
        shutil.copyfile(cls.run_file, os.path.join(build_dir, "run.cpp"))


class DefaultManager(CompileManager):
    run_file = os.path.join(os.path.dirname(__file__), "run.cpp")


class NodeReduceManager(CompileManager):
    run_file = os.path.join(os.path.dirname(__file__), "node_reduce_run.cpp")


class EdgeViewManager(CompileManager):
    run_file = os.path.join(os.path.dirname(__file__), "edge_view.cpp")

    @classmethod
    def form_run_cpp(cls, build_dir):
        header_file = os.path.join(build_dir, "model.h")
        new_run_file = os.path.join(build_dir, "run.cpp")
        make_run_cpp(cls.run_file, new_run_file, header_file)


class ModelZooManager(CompileManager):
    run_file = os.path.join(os.path.dirname(__file__), "model_zoo_run.cpp")

    @classmethod
    def form_run_cpp(cls, build_dir):
        header_file = os.path.join(build_dir, "model.h")
        new_run_file = os.path.join(build_dir, "run.cpp")

        inout_info = get_inout_info(header_file)
        in_macro = inout_info['input']['macro']
        out_macro = inout_info['output'][0]['macro']
        out_size = inout_info['output'][0]['elements'] * 4

        with open(cls.run_file, 'r') as f:
            lines = [line.replace("MODEL_input", in_macro).replace("MODEL_output", out_macro)
                         .replace("160", str(out_size))
                     for line in f.readlines()]

        with open(new_run_file, 'w') as f:
            f.write("".join(lines))


cls_dict = {'default': DefaultManager,
            'node reduce': NodeReduceManager,
            'edge view': EdgeViewManager,
            'model zoo': ModelZooManager}

def select_manager(mode):
    return cls_dict[mode]
