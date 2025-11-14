
from mako.template import Template
from collections import OrderedDict
import os
from . import writer_utils as utils


def print_template_network(
    graph,
    HW_description,
    config_file,
    verbose_level,
    perf_layer,
    app_directory,
    inc_dir_rel,
    src_dir_rel
):
    # Generate the Network management c file.
    tk = OrderedDict([])
    prefix = graph[0].prefix
    tk['prefix'] = prefix
    tk['verbose'] = 'Check' in verbose_level
    tk['sdk'] = HW_description["software development kit"]["name"]
    
    # compute the number of weights tensors
    weights_number = 0
    for nodes in graph:
        if 'FullyConnected' in nodes.name or 'Conv' in nodes.name:
            weights_number += 1
            
    tk['weights_number'] = weights_number
    tk['verbose_level'] = verbose_level
    tk['performance'] = perf_layer
    tk['l1_buffer'] = HW_description["memory"]["L1"]["dimension"] - HW_description["HW specific parameters"]["accelerator core0 stack"] - 7 * HW_description["HW specific parameters"]["accelerator core1-7 stack"]
    tk['master_stack'] = HW_description["HW specific parameters"]["accelerator core0 stack"] 
    tk['slave_stack'] = HW_description["HW specific parameters"]["accelerator core1-7 stack"]
    tk['l2_buffer_size'] = HW_description["memory"]["L2"]["dimension"] - config_file["code reserved space"] 
    
    # working frequencies
    tk['fc_frequency'] = HW_description["core frequency"]
    tk['cl_frequency'] = HW_description["accelerator frequency"]
    if "peripheral frequency" in HW_description:
        tk['periph_frequency'] = HW_description["peripheral frequency"]
    else:
        tk['periph_frequency'] = None
    
    MACs = 0
    file_list_w = []
    list_h = []
    list_name = []
    for i, node in enumerate(graph):
        MACs += node.MACs
        if "Conv" in node.name or "FullyConnected" in node.name:
            file_list_w.append(node.prefixed_name+"_weights.hex")
        list_h.append(node.prefixed_name+".h")
        list_name.append(node.prefixed_name)
    tk['MACs'] = MACs
    tk['files_list'] = utils.print_file_list(file_list_w)
    list_h = list(set(list_h))
    tk['list_h'] = list_h
    tk['func_name'] = list_name
    
    tk['n_inputs'] = graph[0].n_test_inputs
    tk['DORY_HW_graph'] = graph
    
    # debug
    log_str = ""
    for k, v in tk.items():
        try:
            log_str += "// %s %d\n" % (k.ljust(30), v)
        except TypeError:
            try:
                log_str += "// %s %d\n" % (k.ljust(30), v[0])
            except (TypeError, IndexError):
                log_str += "// %s %s\n" % (k.ljust(30), v)
    
    # only render checksum block if all nodes have the required attributes
    render_checksums = (
        graph and
        all(hasattr(node, 'check_sum_in') and node.check_sum_in is not None for node in graph) and
        all(hasattr(node, 'check_sum_out') and node.check_sum_out is not None for node in graph) and
        all(hasattr(node, 'check_sum_w') and node.check_sum_w is not None for node in graph)
    )
    tk['render_checksum'] = render_checksums

    # render the templates with the required informations
    root = os.path.realpath(os.path.dirname(__file__))
    tmpl = Template(
        filename=os.path.join(
            root, 
            "../../Hardware_targets", 
            HW_description["name"], 
            "Templates/network_c_template.c"
        )
    )
    s = tmpl.render(verbose_log=log_str, **tk)
    save_string = os.path.join(app_directory, src_dir_rel, prefix + 'network.c')
    with open(save_string, "w") as f:
        f.write(s)

    tmpl = Template(
        filename=os.path.join(
            root, 
            "../../Hardware_targets", 
            HW_description["name"], 
            "Templates/network_h_template.h"
        )
    )
    s = tmpl.render(verbose_log=log_str, **tk)
    save_string = os.path.join(app_directory, inc_dir_rel, prefix + 'network.h')
    with open(save_string, "w") as f:
        f.write(s)

    tmpl = Template(
        filename=os.path.join(
            root,
            "../../Hardware_targets",
            HW_description["name"],
            "Templates/main_template.c"
        )
    )
    s = tmpl.render(verbose_log=log_str, **tk)
    save_string = os.path.join(app_directory, src_dir_rel, prefix + 'main.c')
    with open(save_string, "w") as f:
        f.write(s)
