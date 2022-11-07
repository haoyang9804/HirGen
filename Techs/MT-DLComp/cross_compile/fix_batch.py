import copy


def fix_model_batch(model, batch_size=1):
    for e in model.graph.input:
        fix_edge_batch(model.graph.input, e, batch_size)
    for e in model.graph.output:
        fix_edge_batch(model.graph.output, e, batch_size)
    for e in model.graph.value_info:
        fix_edge_batch(model.graph.value_info, e, batch_size)


def fix_edge_batch(edge_list, edge, batch_size=1):
    if not hasattr(edge.type.tensor_type, 'shape'):
        return
    new_edge = None
    flag = False
    for i, dim in enumerate(edge.type.tensor_type.shape.dim):
        if 'batch' in dim.dim_param:
            if not flag:
                flag = True
                new_edge = copy.copy(edge)
            new_edge.type.tensor_type.shape.dim[i].dim_value = batch_size
    if flag:
        edge_list.remove(edge)
        edge_list.append(new_edge)
