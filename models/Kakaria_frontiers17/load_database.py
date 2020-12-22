import csv
import json
import re
import sys
import pickle
import itertools
import logging

import path

from cxcircuits.parse_arborization import NeuronArborizationParser

import numpy as np
from pyorient.ogm import Graph, Config

import neuroarch.models as models
import neuroarch.query as query
import neuroarch.nxtools as nxtools

cx_db = 'plocal://localhost:2424/flycircuit'
cx_version = 'Kakaria_frontiers17'
initial_drop = False
model_version = "Kakaria_frontiers17"


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
                'PB','EB','NO', 'no','LAL', 'lal'
                ]

neuropil_name_to_node = {}
for neuropil in CX_Neuropils:
    node = graph.Neuropils.create(name = neuropil, version = cx_version)
    neuropil_name_to_node[neuropil] = node

logger.info('Created Neuropils')

# File names grouped by neuropil in which neurons' presynaptic terminals
# arborize:
data = path.Path('neurons')

# LAL subregions list
LAL_subregions_list = ['RGT', 'RDG', 'RVG', 'RHB']
# lal subregions list
lal_subregions_list = ['LGT', 'LDG', 'LVG', 'LHB']

# NO subregions list
NO_subregions_list = ['(1,R)', '(2,RD)', '(2,RV)','(3,RP)','(3,RM)','(3,RA)']
# no subregions list
no_subregions_list = ['(1,L)', '(2,LD)', '(2,LV)','(3,LP)','(3,LM)','(3,LA)']



neuropil_to_subregions = {'PB': ['L'+str(i) for i in range(1,10)]+['R'+str(i) for i in range(1,10)],
                        'EB': ['L'+str(i) for i in range(1,9)]+['R'+str(i) for i in range(1,9)],#+[str(i) for i in range(1,9)],
                        'LAL': LAL_subregions_list,
                        'lal': lal_subregions_list,
                        'NO': NO_subregions_list,
                        'no': no_subregions_list}


neuropil_to_file_list = {'EB': data.files('eb_lal_pb.csv'),
                         'PB': data.files('pb_eb*.csv')+\
                               data.files('pb_local.csv')
                        }

# File names grouped by neuron family:
family_to_file_list = {'EB-LAL-PB': data.files('eb_lal_pb.csv'),
                       'PB-EB-LAL': data.files('pb_eb_lal.csv'),
                       'PB-EB-NO': data.files('pb_eb_no.csv'),
                       'PB': data.files('pb_local.csv')
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
            # autosynapses are not created.
            if pre_rid != post_rid:
                pre_name = df_nodes.loc[pre_rid]['name']
                post_name = df_nodes.loc[post_rid]['name']
                pre_family = df_nodes.loc[pre_rid]['family']
                post_family = df_nodes.loc[post_rid]['family']
                pre_neuron_parse = parser.parse(pre_name)
                post_neuron_parse = parser.parse(post_name)
                name = pre_name+'->'+ \
                       post_name+'_in_'+ \
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



#############################   Load Model #################################


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
        family = neuron.get_props()['family']
        params = {'initV': -52.,
                  'reset_potential': -72.,
                  'threshold': -45.,
                  # 'resistance': 10000.0,
                  'time_constant': 20.0,
                  'capacitance': 2e-2, #\muF
                  'resting_potential': -52.,
                  'refractory_period': 2.2,
                  'bias_current': 0.0}
        exec_node = graph.LeakyIAFwithRefractoryPeriods.create(name = neuron.name)
        exec_node.update(**params)
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
        q_synapse = query.QueryWrapper.from_objs(graph, [synapse])
        q_post_neuron = q_synapse.gen_traversal_out(
                            ['SendsTo', 'Neuron', 'instanceof'],
                            min_depth = 1)
        post_neuron_bio = q_post_neuron.get_as('obj', edges = False)[0][0]
        post_neuron, _ = q_post_neuron.gen_traversal_in(
                                ['Models', 'AxonHillockModel','instanceof'],
                                min_depth = 1).get_as('obj', edges = False)
        post_neuron = post_neuron[0]

        q_pre_neuron = q_synapse.gen_traversal_in(
                            ['SendsTo', 'Neuron', 'instanceof'],
                            ['Models', 'AxonHillockModel','instanceof'],
                            min_depth = 2)
        pre_neuron = q_pre_neuron.get_as('obj', edges = False)[0][0]
        pre_neuron_LPU = q_pre_neuron.owned_by(
                            cls = 'LPU', version = cx_version).get_as('obj', edges = False)[0][0]
        pre_neuron_bio = q_synapse.gen_traversal_in(
                            ['SendsTo', 'Neuron', 'instanceof'],
                            min_depth = 1).get_as('obj', edges = False)[0][0]

        # params = {}
        pre_family = pre_neuron_bio.get_props()['family']
        post_family = post_neuron_bio.get_props()['family']

        if pre_family == 'EB-LAL-PB':
            params = {'gmax': 0.0175*0.8, 'ar': 0.9, 'ad': 0.15, 'reverse': 0.}
        elif pre_family in ['PB-EB-LAL', 'PB-EB-NO']:
            if post_neuron.name.startswith('EB') and post_neuron.name.split('-')[2].split('/')[1][1]=='9':
                params = {'gmax': 0.0175*0.8, 'ar': 0.9, 'ad': 0.15, 'reverse': 0.}
            else:
                params = {'gmax': 0.0175*0.8/2, 'ar': 0.9, 'ad': 0.15, 'reverse': 0.}

        elif pre_family == 'PB':
            if post_family == 'PB':
                tmp = parser.parse(pre_neuron.name)
                for n in tmp:
                    if 'b' in n['neurite']:
                        pre_regions = n['regions']
                tmp = parser.parse(post_neuron.name)
                for n in tmp:
                    if 's' in n['neurite']:
                        post_regions = n['regions']
                length = len(pre_regions.intersection(post_regions))
                params = {'gmax': 0.0207*1.58*0.8/3*4/length, 'ar': 0.9, 'ad': 0.15, 'reverse': -85.}
            else:
                params = {'gmax': 0.0207*1.58*0.8, 'ar': 0.9, 'ad': 0.15, 'reverse': -85.}
        model_cls = graph.AlphaSynapses

        exec_node = model_cls.create(name = synapse.name)
        graph.Models.create(exec_node, synapse, version = model_version)
        graph.Owns.create(lpu, exec_node)
        exec_node.update(**params)

        graph.SendsTo.create(exec_node, post_neuron)
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
