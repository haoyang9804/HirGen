import re
import os


def get_header_info(header_file):
    with open(header_file, "r") as f:
        lines = [line for line in f.readlines()]
        for i, line in enumerate(lines):
            m = re.match(r"\/\/   Name: \"(A(\d+)(__\d)?)\"", line.strip())
            if m:
                edge_macro = m.group(1)
                break
        for line in lines[i + 1:]:
            m = re.match(r"\/\/   Size: (\d+) \(bytes\)", line.strip())
            if m:
                edge_size = m.group(1)
                break
    return edge_macro, edge_size


def get_inout_info(header_file):
    with open(header_file, 'r') as f:
        lines = [line for line in f.readlines()]
    model_inout = {'input': None, 'output': []}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"// {3}Name: \"(\w+)\"", line)
        if m:
            edge_info = get_edge_info(lines, i)
            info_dict = {'name': edge_info[0], 'shape': edge_info[1],
                         'elements': edge_info[2], 'offset': edge_info[3],
                         'macro': edge_info[4]}
            if len(info_dict['shape']) == 4 and info_dict['shape'][2] > 1:
                model_inout['input'] = info_dict
            else:
                model_inout['output'].append(info_dict)
            i += 5
        else:
            i += 1
    return model_inout


def get_edge_info(lines, start_line_number):
    m = re.match(r"// {3}Name: \"(\w+)\"", lines[start_line_number])
    name = m.group(1)
    m = re.match(r"// {3}Type: float<(\d+)((?: x \d+)*)>", lines[start_line_number + 1])
    edge_shape = tuple([int(m.group(1))] + [int(t) for t in re.findall(r"\d+", m.group(2))])
    m = re.match(r"// {3}Size: (\d+) \(elements\)", lines[start_line_number+2])
    size_elements = int(m.group(1))
    m = re.match(r"// {3}Offset: (\d+) \(bytes\)", lines[start_line_number+4])
    offset = int(m.group(1))
    edge_macro = f"MODEL_{name}"
    return name, edge_shape, size_elements, offset, edge_macro


def make_run_cpp(ori_run_file, new_run_file, header_file):
    with open(ori_run_file, 'r') as f:
        lines = f.readlines()
    edge_macro, edge_size = get_header_info(header_file)
    lines = [line.replace("MODEL_A126", "MODEL_%s" % edge_macro)
                 .replace('262144', edge_size) for line in lines]

    with open(new_run_file, 'w') as f:
        f.write("".join(lines))


def form_edge_diff_cpp(build_dir, ori_run_file):
    header_file = os.path.join(build_dir, "model.h")
    new_run_file = os.path.join(build_dir, "run.cpp")
    make_run_cpp(ori_run_file, new_run_file, header_file)
