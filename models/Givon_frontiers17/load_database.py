import csv
import json
import re
import sys
import pickle
import itertools
import logging

import path

from cx.parse_arborization import NeuronArborizationParser

import numpy as np
import numpy as np
from pyorient.ogm import Graph, Config

import neuroarch.models as models
import neuroarch.query as query
import neuroarch.nxtools as nxtools

cx_db = 'plocal://localhost:2424/flycircuit'
cx_version = 'wild_type'
initial_drop = False
model_version = "Givon_frontiers17"


def convert_set_to_list(a):
    for key in a:
        if isinstance(a[key],set):
            a[key] = list(a[key])
    return a

def convert_complicate_list(a):
    tmp = []
    for region in a['regions']:
        tmp.append(list(region))
    a['regions'] = tmp
    return a

graph = Graph(Config.from_url(cx_db, 'root', 'root',
                              initial_drop=initial_drop))
if initial_drop:
    graph.create_all(models.Node.registry)
    graph.create_all(models.Relationship.registry)
else:
    graph.include(models.Node.registry)
    graph.include(models.Relationship.registry)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('cx')

logger.info('Start to Load Biological Data')

CX_Neuropils = [
                'PB','FB','EB','NO', 'no',
                'BU','bu', 'LAL', 'lal', 'cre', 'CRE'
                ]

neuropil_name_to_node = {}
for neuropil in CX_Neuropils:
    node = graph.Neuropils.create(name = neuropil, version = cx_version)
    neuropil_name_to_node[neuropil] = node

logger.info('Created Neuropils')

# File names grouped by neuropil in which neurons' presynaptic terminals
# arborize:
data = path.Path('neurons')

# FB subregions list
fb_subregions_list = []
for i in range(1, 6):
    for j in range(1, 5):
        fb_subregions_list.append('({},L{})'.format(i,j))
    for j in range(1, 5):
        fb_subregions_list.append('({},R{})'.format(i,j))

for i in range(6, 10):
    fb_subregions_list.append('({})'.format(i))

# LAL subregions list
LAL_subregions_list = ['RGT', 'RDG', 'RVG', 'RHB']
# lal subregions list
lal_subregions_list = ['LGT', 'LDG', 'LVG', 'LHB']

# NO subregions list
NO_subregions_list = ['(1,R)', '(2,RD)', '(2,RV)','(3,RP)','(3,RM)','(3,RA)']
# no subregions list
no_subregions_list = ['(1,L)', '(2,LD)', '(2,LV)','(3,LP)','(3,LM)','(3,LA)']



neuropil_to_subregions = {'BU': ['R{}'.format(i) for i in range(1,81)],
                        'bu': ['L{}'.format(i) for i in range(1,81)],
                        'PB': ['L'+str(i) for i in range(1,10)]+['R'+str(i) for i in range(1,10)],
                        'EB': ['L'+str(i) for i in range(1,9)]+['R'+str(i) for i in range(1,9)],#+[str(i) for i in range(1,9)],
                        'FB': fb_subregions_list,
                        'LAL': LAL_subregions_list,
                        'lal': lal_subregions_list,
                        'NO': NO_subregions_list,
                        'no': no_subregions_list,
                        # 'IB': ['R'],
                        # 'ib': ['L'],
                        # 'PS': ['R'],
                        # 'ps': ['L'],
                        # 'WED': ['R'],
                        # 'wed': ['L'],
                        'CRE': ['RRB', 'RCRE'],
                        'cre': ['LRB', 'LCRE']}


neuropil_to_file_list = {'BU': data.files('bu_eb_r.csv'),
                         'bu': data.files('bu_eb_l.csv'),
                         'FB': data.files('fb_local.csv'),
                         'EB': data.files('eb_lal_pb.csv'),
                         'PB': data.files('pb*.csv')
                        }

# File names grouped by neuron family:
family_to_file_list = {'BU-EB': data.files('bu_eb_*.csv'),
                       'FB': data.files('fb_local.csv'),
                       'EB-LAL-PB': data.files('eb_lal_pb.csv'),
                       'PB-EB-LAL': data.files('pb_eb_lal.csv'),
                       'PB-EB-NO': data.files('pb_eb_no.csv'),
                       'PB-FB-CRE': data.files('pb_fb_cre.csv'),
                       'PB-FB-LAL': data.files('pb_fb_lal*.csv'),
                       'PB-FB-NO': data.files('pb_fb_no*.csv'),
                       'PB': data.files('pb_local.csv'),
                       }

# Map file names to neuron family:
file_to_family = {}
for family in family_to_file_list:
    for file_name in family_to_file_list[family]:

        # Sanity check against typos in file list:
        if file_name in file_to_family:
            raise RuntimeError('duplicate file name')
        file_to_family[file_name] = family

# Parse labels into neuron data (i.e., each neuron associated with its label)
# and arborization data (i.e., each arborization associated with its originating
# label):
parser = NeuronArborizationParser()
# neuropil_data = [{'name': neuropil} for neuropil in neuropil_to_file_list]
# for retreiving node by name
subregion_name_to_node = {}

for neuropil in CX_Neuropils:
    for subregion in neuropil_to_subregions[neuropil]:
        node = graph.Subregions.create(name = '{}/{}'.format(neuropil, subregion))
        node.update(**{'neuropil': neuropil})
        subregion_name_to_node[frozenset([neuropil,subregion])] = node
        graph.Owns.create(neuropil_name_to_node[neuropil], node)

logger.info('Created Subregions')

neuron_name_to_node = {}
arbor_name_to_node = {}
terminal_name_to_node = {}
tract_dict = {}
neuron_rid_to_neuropil = {}

for neuropil in neuropil_to_file_list.keys():
    for file_name in neuropil_to_file_list[neuropil]:
        with open(file_name, 'r') as f:
            r = csv.reader(f, delimiter=' ')
            for row in r:
                d = {'name': row[0], 'family': file_to_family[file_name]}

                # Add 'neuropil' attrib to each neuron data entry to enable ETL to
                # link each Neuron node to the appropriate Neuropil node:
#                 neuron_data.append(d)
                node = graph.Neurons.create(name = d['name'])
                node.update(**d)
                neuron_name_to_node[d['name']] = node
                graph.Owns.create(neuropil_name_to_node[neuropil], node)
                neuron_rid_to_neuropil[node._id] = neuropil_name_to_node[neuropil]
                try:
                    tmp = parser.parse(row[0])
                except Exception as e:
                    print(file_name, row[0])
                    raise e

                axon_neuropils = set([a['neuropil'] for a in tmp \
                                  if 'b' in a['neurite']])
                # dendrite_neuropils = [a['neuropil'] for a in tmp \
                #                       if 's' in a['neurite']]
                # if len(dendrite_neuropils) > 1:
                #     raise ValueError(
                #         "Neuron {} has dendrite in more than one neuropil".format(node.name))
                # else:
                # dendrite_neuropil = dendrite_neuropils[0]
                dendrite_neuropil = neuropil
                for axon_neuropil in axon_neuropils:
                    if dendrite_neuropil != axon_neuropil:
                        neuropil_pair = frozenset(
                                        [dendrite_neuropil, axon_neuropil])
                        if neuropil_pair not in tract_dict:
                            tract_name = '-'.join(sorted(list(neuropil_pair)))
                            tract = graph.Tracts.create(name = tract_name,
                                                        version = cx_version,
                                                        neuropils = set(neuropil_pair))
                            tract_dict[neuropil_pair] = tract
                        graph.Owns.create(tract_dict[neuropil_pair], node)

                # Add 'neuron' attrib to each arborization data entry to enable
                # ETL to link each ArborizationDAta node to the appropriate
                # Neuron node:
                for a in tmp:
                    a['neuron'] = row[0]
#                     arbor_data.append(a)
                    node = graph.ArborizationDatas.create(name='arbor')
#                     print(a)
#                     print(type(a['regions']))
                    a = convert_set_to_list(a)
                    complicate_list = ['FB','NO','no']
                    if a['neuropil'] in complicate_list:
                        a = convert_complicate_list(a)
                    # node.update(**a)
                    # graph.Owns.create(neuron_name_to_node[a['neuron']], node)

                    for b in a['regions']:
                        # terminal = {}
                        # terminal['name'] = str(b).replace("'","")
                        # terminal['neurite'] = a['neurite']
                        # terminal['neuropil'] = a['neuropil']
                        # terminal['neuron'] = a['neuron']
                        # # terminal_data.append(terminal)
                        # node = graph.NeuronTerminals.create(name=terminal['name'])
                        # terminal = convert_set_to_list(terminal)
                        # node.update(**terminal)
                        # graph.Owns.create(neuron_name_to_node[a['neuron']], node)
                        if isinstance(b, list):
                            subregion_name = '({})'.format(','.join(b))
                        else:
                            subregion_name = b
                        graph.ArborizesIn.create(
                            neuron_name_to_node[a['neuron']],
                            subregion_name_to_node[frozenset([a['neuropil'],subregion_name])],
                            kind = a['neurite'])

logger.info('Created Neurons and Tracts')

q = query.QueryWrapper(graph,
                       query.QueryString(
                         "select from Neuropil where name in {} and version='{}'".format(
                              "['"+"','".join(CX_Neuropils)+"']", cx_version),
                         'sql'))
q_subregions = q.owns(1,cls="Subregion")
subregions, _ = q_subregions.get_as('obj', edges = False)

for subregion in subregions:
    qq = query.QueryWrapper.from_objs(graph, [subregion])
    subregion_neuropil = qq.owned_by(
                            cls = 'Neuropil',
                            version = cx_version).get_as('obj', edges = False)[0][0]
    g = qq.gen_traversal_in(['ArborizesIn', 'Neuron', 'instanceof'])
    df_nodes, df_edges = g.get_as('df')
    b = []
    s = []
    for ind in df_edges.index:
        neuron_rid = df_edges.loc[ind]['out']
        for kind in df_edges.loc[ind]['kind']:
            if kind == 's':
                s.append(neuron_rid)
            elif kind == 'b':
                b.append(neuron_rid)

    for post_rid in s:
        post_neuropil = neuron_rid_to_neuropil[post_rid]
        for pre_rid in b:
            if pre_rid != post_rid:
                name = df_nodes.loc[pre_rid]['name']+'->'+ \
                       df_nodes.loc[post_rid]['name']+'_in_'+ \
                       subregion.name#.replace('/','-')
                syn = graph.Synapses.create(name = name)

                graph.SendsTo.create(graph.get_element(pre_rid), syn)
                graph.SendsTo.create(syn, graph.get_element(post_rid))
                graph.Owns.create(subregion, syn)
                graph.Owns.create(post_neuropil, syn)
                if subregion_neuropil.name != post_neuropil.name:
                    pre_neuropil = neuron_rid_to_neuropil[pre_rid]
                    if pre_neuropil.name != post_neuropil.name:
                        neuropil_pair = frozenset([pre_neuropil.name, post_neuropil.name])
                        if neuropil_pair not in tract_dict:
                            tract_name = '-'.join(sorted(list(neuropil_pair)))
                            tract = graph.Tracts.create(name = tract_name,
                                                        version = cx_version,
                                                        neuropils = set(neuropil_pair))
                            tract_dict[neuropil_pair] = tract
                            logger.info('created new tract {}: {}'.format(tract.name, syn.name))
                        graph.Owns.create(tract_dict[neuropil_pair],
                                          graph.get_element(pre_rid))

logger.info('Created Synapses')

logger.info('Biological Data Loaded')


######################### Load Model ##############################

q = query.QueryWrapper(graph,
                       query.QueryString(
                         "select from Neuropil where name in {} and version='{}'".format(
                              "['"+"','".join(CX_Neuropils)+"']", cx_version),
                         'sql'))
neuropils, _ = q.get_as('obj', edges = False)

lpu_dict = {}   # map LPU name to LPU nodes
lpu_port_counter = {}

for neuropil in neuropils:
    # create LPUs
    lpu = graph.LPUs.create(name = neuropil.name,
                            version = model_version)
    graph.Models.create(lpu, neuropil, version = model_version) # need to add version?
    # create LPU Interfaces
    interface = graph.Interfaces.create(name = neuropil.name)
    graph.Owns.create(lpu, interface)
    lpu_dict[lpu.name] = {'LPU': lpu, 'Interface': interface}

    lpu_port_counter[lpu.name] = {'out': itertools.count(),
                                  'in': itertools.count()}

logger.info('Created LPUs')


CX_Tracts = list(set(['-'.join(sorted(a))for a in itertools.product(CX_Neuropils, CX_Neuropils)])\
            - set(['-'.join(a) for a in zip(CX_Neuropils, CX_Neuropils)]))
q_tract = query.QueryWrapper(graph,
                       query.QueryString(
                         "select from Tract where name in {} and version='{}'".format(
                              "['"+"','".join(CX_Tracts)+"']", cx_version),
                         'sql'))
tracts, _ = q_tract.get_as('obj', edges = False)
pattern_dict = {}

for tract in tracts:
    linked_neuropils = sorted(list(tract.neuropils))
    pattern_pair = frozenset(linked_neuropils)

    pattern = graph.Patterns.create(name = tract.name,
                                    version = model_version,
                                    LPUs = tract.neuropils)
    graph.Models.create(pattern, tract, version = model_version)

    int_0 = graph.Interfaces.create(
        name = '{}/{}'.format(tract.name, linked_neuropils[0]))
    int_1 = graph.Interfaces.create(
        name = '{}/{}'.format(tract.name, linked_neuropils[1]))
    graph.Owns.create(pattern, int_0)
    graph.Owns.create(pattern, int_1)
    pattern_dict[pattern_pair] = {'pattern': pattern,
                                  linked_neuropils[0]: int_0,
                                  linked_neuropils[1]: int_1}

logger.info('Created Patterns')

for neuropil in neuropils:
    neuropil_name = neuropil.name
    lpu = lpu_dict[neuropil_name]['LPU']
    q_neuropil = query.QueryWrapper.from_objs(graph, [neuropil])
    q_neurons = q_neuropil.traverse_owns(max_levels = 1, cls = 'Neuron')
    neurons, _ = q_neurons.get_as('obj', edges = False)
    for neuron in neurons:
        exec_node = graph.LeakyIAFs.create(name = neuron.name)
        graph.Models.create(exec_node, neuron, version = model_version)
        graph.Owns.create(lpu, exec_node)

        q_neuron = query.QueryWrapper.from_objs(graph, [neuron])
        synapse_loc, _ = q_neuron.gen_traversal_out(
                            ['SendsTo', 'Synapse', 'instanceof'],
                            min_depth = 1).owned_by(
                                cls = 'Neuropil',
                                version = cx_version).get_as('obj', edges = False)
        if any([loc.name != neuropil_name for loc in synapse_loc]):
            sel_i = '/{}/out/spk/{}'.format(
                        neuropil_name,
                        next(lpu_port_counter[neuropil_name]['out']))
            port_i = graph.Ports.create(selector = sel_i,
                                        port_io = 'out',
                                        port_type = 'spike',
                                        neuron = neuron.name)
            # Interface nodes must own the new Port nodes:
            graph.Owns.create(lpu_dict[neuropil_name]['Interface'], port_i)
            graph.SendsTo.create(exec_node, port_i)
            for loc in synapse_loc:
                if loc.name != neuropil_name:
                    sel_j = '/{}/in/spk/{}'.format(
                            loc.name, next(lpu_port_counter[loc.name]['in']))
                    port_j = graph.Ports.create(selector = sel_j,
                                                port_io = 'in',
                                                port_type = 'spike',
                                                neuron = neuron.name)
                    graph.Owns.create(lpu_dict[loc.name]['Interface'], port_j)

                    pattern_pair = frozenset([neuropil_name, loc.name])
                    pattern = pattern_dict[pattern_pair]['pattern']
                    int_0 = pattern_dict[pattern_pair][neuropil_name]
                    int_1 = pattern_dict[pattern_pair][loc.name]

                    pat_port_i = graph.Ports.create(selector = port_i.selector,
                                                    port_io = 'in',
                                                    port_type= port_i.port_type,
                                                    neuron = neuron.name)
                    graph.Owns.create(int_0, pat_port_i)

                    pat_port_j = graph.Ports.create(selector = port_j.selector,
                                                    port_io = 'out',
                                                    port_type = port_j.port_type,
                                                    neuron = neuron.name)
                    graph.Owns.create(int_1, pat_port_j)

                    graph.SendsTo.create(pat_port_i, pat_port_j)
                    graph.SendsTo.create(port_i, pat_port_i)
                    graph.SendsTo.create(pat_port_j, port_j)

logger.info('Created Neuron Models and Ports')

for neuropil in neuropils:
    neuropil_name = neuropil.name
    lpu = lpu_dict[neuropil_name]['LPU']
    q_neuropil = query.QueryWrapper.from_objs(graph, [neuropil])
    synapses, _ = q_neuropil.traverse_owns(
                        max_levels = 1, cls = 'Synapse').get_as('obj', edges = False)
    for synapse in synapses:
        exec_node = graph.AlphaSynapses.create(name = synapse.name)
        graph.Models.create(exec_node, synapse, version = model_version)
        graph.Owns.create(lpu, exec_node)
        q_synapse = query.QueryWrapper.from_objs(graph, [synapse])
        post_neuron, _ = q_synapse.gen_traversal_out(
                            ['SendsTo', 'Neuron', 'instanceof'],
                            min_depth = 1).gen_traversal_in(
                                ['Models', 'AxonHillockModel','instanceof'],
                                min_depth = 1).get_as('obj', edges = False)
        graph.SendsTo.create(exec_node, post_neuron[0])

        q_pre_neuron = q_synapse.gen_traversal_in(
                            ['SendsTo', 'Neuron', 'instanceof'],
                            ['Models', 'AxonHillockModel','instanceof'],
                            min_depth = 2)
        pre_neuron = q_pre_neuron.get_as('obj', edges = False)[0][0]
        pre_neuron_LPU = q_pre_neuron.owned_by(
                            cls = 'LPU', version = cx_version).get_as('obj', edges = False)[0][0]
        if pre_neuron_LPU.name == neuropil_name:
            graph.SendsTo.create(pre_neuron, exec_node)
        else:
            q_port = q_pre_neuron.gen_traversal_out(
                        ['SendsTo', 'Port'],
                        ['SendsTo', 'Port'],
                        ['SendsTo', 'Port'],
                        ['SendsTo', 'Port'],
                        min_depth = 4)
            q_interface = q_port.owned_by(cls = 'Interface')
            df, _ = q_interface.get_as('df', edges = False)
            rid = df.index[df['name'] == neuropil_name].tolist()
            if len(rid) > 1:
                raise ValueError("Multiple Ports Detected")
            else:
                port = (q_port+q_interface).edges_as_objs[0].inV()


            logger.info('created link between %s of %s: %s' % (
                            port.selector, pre_neuron.name, exec_node.name))
            graph.SendsTo.create(port, exec_node)

logger.info('Created Synapse Models')


##################  Load Parameters ######################################



def leaky_iaf_params(lpu, extern):
    """
    Generate LeakyIAF params.
    """
    k = 1000
    assert isinstance(extern, bool)
    if lpu == 'BU' or lpu == 'bu':
        return {'extern': extern,
                'initV': -0.06 * k,
                'reset_potential': -0.0675489770451* k,
                #'Vt': -0.0251355161007,
                'threshold': -0.001* k,
                'resistance': 1.02445570216* k,
                'capacitance': 0.0669810502993,
                'resting_potential': 0.0* k}
    elif lpu == 'EB':
        return {'extern': extern,
                'initV': -0.06* k,
                'reset_potential': -0.0675489770451* k,
                #'Vt': -0.0251355161007,
                'threshold': -0.001* k,
                'resistance': 1.02445570216* k,
                'capacitance': 0.0669810502993,
                'resting_potential': 0.0* k}
    elif lpu == 'FB':
        return {'extern': extern,
                'initV': -0.06* k,
                'reset_potential': -0.0675489770451* k,
                #'Vt': -0.0251355161007,
                'threshold': -0.001* k,
                'resistance': 1.02445570216* k,
                'capacitance': 0.0669810502993,
                'resting_potential': 0.0* k}
    elif lpu == 'PB':
        return {'extern': extern,
                'initV': -0.06* k,
                'reset_potential': -0.07* k,
                #'Vt': -0.0251355161007,
                'threshold': -0.00001* k,
                'resistance': 0.25* k,
                'capacitance': 0.5,
                'resting_potential': 0.0* k}
    elif lpu in ['LAL', 'lal']:
        return {'extern': extern,
                'initV': -0.06* k,
                'reset_potential': -0.07* k,
                #'Vt': -0.0251355161007,
                'threshold': -0.00001* k,
                'resistance': 0.25* k,
                'capacitance': 0.5,
                'resting_potential': 0.0* k}
    elif lpu in ['NO', 'no']:
        return {'extern': extern,
                'initV': -0.06* k,
                'reset_potential': -0.07* k,
                #'Vt': -0.0251355161007,
                'threshold': -0.00001* k,
                'resistance': 0.25* k,
                'capacitance': 0.5,
                'resting_potential': 0.0* k}
    else:
        raise ValueError('unrecognized LPU name')

def alpha_synapse_params(lpu):
    """
    Generate AlphaSynapse params.
    """
    k = 1000
    s = 1e-3#0.001
    s1 = 0.001
    if lpu == 'BU' or lpu == 'bu':
        return {'conductance': True,
                'ad': 0.16,
                'ar': 0.11,
                'gmax': 1e-3*1,
                'reverse': -65.}
    elif lpu == 'EB':
        return {'conductance': True,
                'ad': 0.16,
                'ar': 0.11,
                'gmax': 3e-3,
                'reverse': 65.}
    elif lpu == 'FB':
        return {'conductance': True,
                'ad': 0.16,
                'ar': 0.11,
                'gmax': 1e-2,
                'reverse': 65.}
    elif lpu == 'PB':
        return {'conductance': True,
                'ad': 0.19,
                'ar': 0.11,
                'gmax': 2e-3*10,
                'reverse': 65.}
    elif lpu in ['LAL', 'lal']:
        return {'conductance': True,
                'ad': 0.19,
                'ar': 0.11,
                'gmax': 2e-3,
                'reverse': 65.}
    elif lpu in ['NO', 'no']:
        return {'conductance': True,
                'ad': 0.19,
                'ar': 0.11,
                'gmax': 2e-3,
                'reverse': 65.}
    elif lpu in ['CRE', 'cre']:
        return {'conductance': True,
                'ad': 0.19,
                'ar': 0.11,
                'gmax': 2e-3,
                'reverse': 65.}
    else:
        raise ValueError('unrecognized LPU name')



# Get all LeakyIAF/AlphaSynapse nodes in each LPU:

lpu_to_query = {}
for lpu in CX_Neuropils:
    logger.info('retrieving LeakyIAF/AlphaSynapse nodes for LPU %s' % lpu)
    lpu_node = graph.LPUs.query(name = lpu,
                                version = model_version).one() #TODO
    lpu_to_query[lpu] = lpu_node.owns(1, cls = ['LeakyIAF', 'AlphaSynapse'])

# Assign parameters:
for lpu in CX_Neuropils:
    for n in lpu_to_query[lpu].nodes_as_objs:
        if isinstance(n, models.LeakyIAF):
            logger.info('assigning params to %s LeakyIAF %s' % (lpu, n.name))
            n.update(**leaky_iaf_params(lpu, True))
        elif isinstance(n, models.AlphaSynapse):
            c = graph.LeakyIAFs.query(name=n.name.split('_in_')[0].split('->')[0]).all()
            for cc in c:
                for b in cc.in_('Owns'):
                    if isinstance(b, models.LPU):
                        if b.version == model_version:
                            logger.info('assigning params to %s AlphaSynapse %s' % (lpu, n.name))
                            n.update(**alpha_synapse_params(b.name))
                            break

# Rerun queries to fetch updated data:
for lpu in CX_Neuropils:
    logger.info('rerunning query for LPU %s to fetch updated data' % lpu)
    lpu_to_query[lpu].execute(True, True)
