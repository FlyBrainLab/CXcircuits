
import itertools
import json
from copy import deepcopy
import time
import os, sys
from collections import Counter

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import matplotlib.cm as cm
from matplotlib import animation
import matplotlib.image as mpimg

from .parse_arborization import NeuronArborizationParser
from .add_neuron import generate_svg
from parsimonious.exceptions import ParseError

def na_to_nx(x):
    """Builds a NetworkX graph using processed data from NeuroArch.
    # Arguments:
        x (dict): File url.
    # Returns:
        g (NetworkX MultiDiGraph): A MultiDiGraph instance with the circuit graph.
    """
    g = nx.MultiDiGraph()
    g.add_nodes_from(x['nodes'].items())
    for pre,v,attrs in x['edges']:
        g.add_edge(pre, v, **attrs)
    return g

def nx_data_to_graph(data):
    g = nx.MultiDiGraph()
    g.add_nodes_from(list(data['nodes'].items()))
    for u, v, d in data['edges']:
        g.add_edge(u, v, **d)
    return g

def subregion_conversion(name):
    neuropil, region = name.split('/')
    if neuropil == 'EB':
        if region[0] not in ['L', 'R']:
            if region == '1':
                return ['EB/L1', 'EB/R1']
            elif region == '2':
                return ['EB/R2', 'EB/R3']
            elif region == '3':
                return ['EB/R4', 'EB/R5']
            elif region == '4':
                return ['EB/R6', 'EB/R7']
            elif region == '5':
                return ['EB/R8', 'EB/L8']
            elif region == '6':
                return ['EB/L7', 'EB/L6']
            elif region == '7':
                return ['EB/L5', 'EB/L4']
            elif region == '8':
                return ['EB/L2', 'EB/L3']
        else:
            return [name]
    else:
        return [name]

class Subregion_State(object):
    def __init__(self, subregions, active = None):
        """
        Storage and operations on aspects related to subregions.

        Parameters
        ----------
        subregions: list of str or iterables
                    names of subregions
        active: list, set, None
                name of the subregion that are active at the time this object
                is created. If None, all subregions will be set to active
        """
        self.subregions = list(subregions)
        if active is None:
            self.active_subregion = set(subregions)
        else:
            self.active_subregion = set(active)
        self.inactive_subregion = set(self.subregions) - \
                                  self.active_subregion
        self.all_synapses = {\
                    name: set() for name \
                    in subregions}
        self.disabled_synapses = {\
                    name: set() for name \
                    in subregions}

    def load_synapses(self, subregion, synapses):
        self.all_synapses[subregion].update(synapses)

    def enable_synapse(self, synapse):
        subregion = self.get_synapse_subregion(synapse)
        num_activated_synapse = len(self.all_synapses[subregion])
        self.disabled_synapses[subregion].remove(synapse)
        self.all_synapses[subregion].add(synapse)
        if num_activated_synapse == 0:
            self.inactive_subregion.remove(subregion)
            self.active_subregion.add(subregion)
            return subregion
        else:
            return None

    def disable_synapse(self, synapse):
        subregion = self.get_synapse_subregion(synapse)
        num_activated_synapse = len(self.all_synapses[subregion])
        self.disabled_synapses[subregion].add(synapse)
        self.all_synapses[subregion].remove(synapse)
        if len(self.all_synapses[subregion]) == 0:
            self.active_subregion.remove(subregion)
            self.inactive_subregion.add(subregion)
            return subregion
        else:
            return None

    def get_synapse_subregion(self, synapse):
        return synapse.split('_in_')[1]

    def disable_subregion(self, subregion):
        self.disabled_synapses[subregion].update(self.all_synapses)
        self.all_synapses[subregion] = set()
        self.active_subregion.remove(subregion)
        self.inactive_subregion.add(subregion)



class Query(object):
    def __init__(self, fbl):
        """
        Initialize NA query functions

        Parameters
        ----------
        fbl: flybrainlab.Client
             flybrainlab client object
        """
        self.fbl = fbl

    def na_query(self, query_list, format = 'nx'):
        """
        make querying a little easy to do by just specifying the query_list
        """
        inp = {"query": query_list,
               "format": format,
               "user": self.fbl.client._async_session._session_id,
               "server": self.fbl.naServerID
               }
        res = self.fbl.client.session.call('ffbo.processor.neuroarch_query',
                                           inp)
        data = res['success']['result']['data']
        return data


class CX_Constructor(object):
    def __init__(self, fbl, initial_model_version = "Givon_frontiers17"):
        """
        Initialize the CX Model based on a Model version

        Parameters
        ----------
        fbl: flybrainlab.Client
             flybrainlab client object
        initial_model_version: str
                               name of model version

        Properties
        ----------
        fbl: a copy of FBL environment
        querier: query utilities.
        parser: NeuronArborizationParser.
        new_model_initialization: whether a new model is initialize
        data: all data related to the model version retrieved from Neuroarch.
        correlate_list: Neu3D and NeuGFX correlate list.
        rid_dict: a dictionary whose keys are classes and values are dictionary.
                  of object names as keys and their rids as values.
        subregion_arborization: a dictionary where keys are subregion names
                                and values are dictionaries where key 's' holds
                                a set of rids of neurons that has dendrites in
                                the subregion and key 'b' holds a set of rids of
                                neurons that has axon terminals in the subregion.
        subregions: state of subregions
        config: diagram configuration dictionary



        """
        self.fbl = fbl
        #self.fbl.executeNLPquery('remove neurons')
        self.fbl.experimentWatcher = self
        self.initial_model_version = initial_model_version

        self.querier = Query(self.fbl)

        CX_Neuropils = ['PB','EB','BU','bu','FB','NO', 'no', 'LAL', 'lal', 'CRE', 'cre', \
                        'IB', 'ib', 'PS', 'ps', 'WED', 'wed']

        self.CX_LPUs = self._get_model_version(CX_Neuropils)
        # ['PB','FB','EB','NO', 'no','BU','bu', 'LAL', 'lal', 'cre', 'CRE']
        self.CX_Patterns = list(set(['-'.join(sorted(a)) for a in \
                                itertools.product(self.CX_LPUs, self.CX_LPUs)])\
                         - set(['-'.join(a) for a in \
                                zip(self.CX_LPUs, self.CX_LPUs)]))

        # self._load_models()
        # self._load_modeled_neuropils()

        # state variables that need update when neurons are added/removed:
        # self.data
        # self.rid_dict
        # self.subregion_arborization
        # self.added

        # loading all data associated with the specified model
        # create parser
        self.parser = NeuronArborizationParser()

        self.new_model_initialization = False

        self._load_data()

        # produce a dict of rid for different types of objects, including
        # 'Neuropil', 'Tract', 'LPU', 'Pattern', 'Neuron', 'Synapse',
        # 'Subregion', 'Interface', 'NeuronModel', 'SynapseModel', 'Ports'.
        self._organize_rid_dict()

        # create a dictionary specifying
        # for each subregion, what rid of neurons that arborizes in
        # this subregion. e.g.,
        # self.subregion_arborization['EB/1']={'b': [...], 's': [...]}
        self._subregion_innervation()

        self._load_morphology()

        self._load_svg()

        # initialize the configuration for CX diagram in NeuGFX
        #self._initialize_diagram_config()
        self.test = []

    def _load_modeled_neuropils(self):
        query_list = [
                {"action": {"method":
                    {"query":{"name": self.CX_LPUs,
                              "version": self.initial_model_version}}},
                 "object":{"class":"LPU"}},
                {"action": {"method":{
                    "gen_traversal_out":{
                        "min_depth":0,
                        "pass_through":["Models", "Neuropil","instanceof"]}}},
                 "object":{"memory":0}},
                {"action": {"method":{"traverse_owns":{}}},
                 "object":{"memory":0}},
                ]
        data = self.querier.na_query(query_list)
        self.neuropils = nx_data_to_graph(data)

        query_list = [
                {"action": {"method":
                    {"query":{"name": self.CX_Patterns,
                              "version": self.initial_model_version}}},
                 "object":{"class":"Pattern"}},
                {"action": {"method":{"gen_traversal_out":{"min_depth":1, "pass_through":["Models", "Tract","instanceof"]}}},
                 "object":{"memory":0}},
                {"action": {"method":{"traverse_owns":{}}},
                 "object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":2}}},"object":{"memory":0}},
              ]
        data = self.querier.na_query(query_list)
        self.tracts = nx_data_to_graph(data)

    def _get_model_version(self, CX_Neuropils):
        query_list = [
                {"action": {"method":
                    {"query":{"name": CX_Neuropils}}},
                 "object":{"class":"LPU"}},
                ]
        data = self.querier.na_query(query_list)
        return list(set([v['name'] for k,v in data['nodes'].items() \
                         if v.get('version', None) == self.initial_model_version]))

    def _load_svg(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(dir_path, '..', 'models',
                                '{}'.format(self.initial_model_version),
                                'diagram.svg')

        mycode = """
                window.CXDiagramName = '{}';
                """.format(self.initial_model_version)
        self.fbl.tryComms({'widget':'GFX','messageType': 'eval',
                           'data': {'data': mycode, 'name': 'CustomCXInit'}})

        with open(filename, 'r') as file:
            data = file.read()
        self.fbl.tryComms({'widget':'GFX',
                           'messageType': 'loadCircuitFromString',
                           'data': {'string': data, 'name': self.initial_model_version}})
        time.sleep(1)

        self.fbl.tryComms({'messageType': 'loadJS', 'widget':'GFX',
                           'data': 'data/FBLSubmodules/onCXLoad.js'})
        time.sleep(1)

    def _load_morphology(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        correlate_file = os.path.join(dir_path, '..', 'models',
                                      '{}'.format(self.initial_model_version),
                                      'flycircuit_correlates.json')
        with open(correlate_file, 'r') as f:
            correlate = json.load(f)
        corr = {i: [[],[]] for i in set(correlate[0])}
        for i, n in enumerate(correlate[0]):
            corr[n][0].append(correlate[1][i])
            corr[n][1].append(correlate[2][i])
        neurons_loaded = list(self.rid_dict['Neuron'].keys())
        to_send_list = [[],[],[]]
        for neuron in neurons_loaded:
            try:
                names = corr.get(neuron)
                to_send_list[0].extend([neuron]*len(names[0]))
                to_send_list[1].extend(names[0])
                to_send_list[2].extend(names[1])
            except:
                pass
        self.correlate_list = to_send_list
        a = json.dumps(to_send_list)
        mycode = "window.correlates_file = " + a + ";";
        self.fbl.tryComms({'widget':'GFX','messageType': 'eval',
                           'data': {'data': mycode, 'name': 'CXOverride1'}})
        # mycode = "var new_correlates = " + a + ";";
        # mycode += """
        #         window.bioMatches = new_correlates;
        #         window.bioWorkspace = [];
        #         for (var i = 0; i < bioMatches[1].length; i++) {
        #             window.bioWorkspace[bioMatches[1][i]] = true;
        #         }
        #         window.renewCircuit();
        #         window.reloadNeurons3D();
        #     """
        # self.fbl.tryComms({'widget':'GFX','messageType': 'eval',
        #                    'data': {'data': mycode, 'name': 'CXOverride1'}})
        time.sleep(5)

    def _load_models(self):
        """
        Not Used
        """
        query_list = [
                {"action": {"method":
                    {"query":{"name": self.CX_LPUs,
                              "version": self.initial_model_version}}},
                 "object":{"class":"LPU"}},
                {"action":{"method":
                    {"query":{"name": self.CX_Patterns,
                              "version": self.initial_model_version}}},
                 "object":{"class":"Pattern"}},
                {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                {"action": {"method":{"traverse_owns":{}}},
                 "object":{"memory":0}}
              ]
        data = self.querier.na_query(query_list)
        # self.models = nx_data_to_graph(data)
        # self.LPUs = {lpu: na_to_nx(v) for lpu, v in data['LPU'].items()}
        # self.Patterns = {p: na_to_nx(v) for p, v in data['Pattern'].items()}

    def _load_data(self):
        # get LPUs and the Neuropils, and objects they own,
        # including the models relation between pairs of them
        query_list = [
                {"action": {"method":
                    {"query":{"name": self.CX_LPUs,
                              "version": self.initial_model_version}}},
                 "object":{"class":"LPU"}},
                {"action": {"method":{
                    "gen_traversal_out":{
                        "min_depth":0,
                        "pass_through":["Models", "Neuropil","instanceof"]}}},
                 "object":{"memory":0}},
                {"action": {"method":{"traverse_owns":{}}},
                 "object":{"memory":0}},
                ]
        data1 = self.querier.na_query(query_list)

        # get Tracts and objects they own, plus the Patterns that model them
        query_list = [
                {"action": {"method":
                    {"query":{"name": self.CX_Patterns,
                              "version": self.initial_model_version}}},
                 "object":{"class":"Pattern"}},
                {"action": {"method":{
                        "gen_traversal_out":{
                            "min_depth":1,
                            "pass_through":["Models", "Tract","instanceof"]}}},
                 "object":{"memory":0}},
                {"action": {"method":{"traverse_owns":{}}},
                 "object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":2}}},"object":{"memory":0}},
              ]
        data2 = self.querier.na_query(query_list)

        # get interface of LPUs and Patterns, traverse own all.
        # to obtain the relation between ports.
        query_list = [
                {"action": {"method":
                    {"query":{"name": self.CX_LPUs,
                              "version": self.initial_model_version}}},
                 "object":{"class":"LPU"}},
                {"action":{"method":
                    {"query":{"name": self.CX_Patterns,
                              "version": self.initial_model_version}}},
                 "object":{"class":"Pattern"}},
                {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                {"action": {"method":{
                    "gen_traversal_out":{
                        "min_depth":1,
                        "pass_through":["Owns", "Interface","instanceof"]}}},
                 "object":{"memory":0}},
                {"action": {"method":{"traverse_owns":{}}},
                 "object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":3}}},"object":{"memory":0}},
              ]
        data3 = self.querier.na_query(query_list)

        self.data = nx.compose_all([nx_data_to_graph(data1),
                                    nx_data_to_graph(data2),
                                    nx_data_to_graph(data3)])

    def _organize_rid_dict(self):
        """
        Produce a dictionary with keys of main objects
        The values are lists of rid values of corresponding nodes.
        """
        objects_of_interest = ['Neuropil', 'Tract', 'LPU', 'Pattern',
                               'Neuron', 'Synapse', 'Subregion',
                               'Interface']
        self.rid_dict = {
            tlo: {
                  v['name']: k for k, v in self.data.nodes(data=True) \
                  if v['class'] == tlo}
            for tlo in objects_of_interest}

        self.rid_dict.update({
            'Port': {frozenset([v['selector'],v['port_io']]): k \
            for k, v in self.data.nodes(data=True) \
            if v['class'] == 'Ports'}
        })

        # find the NeuronModel and SynapseModels by tracing the Models Relation
        # from Neurons and Synapses.
        neuron_model_dict = {}
        for neuron_name, rid in self.rid_dict['Neuron'].items():
            model_rid = self.find_predecessor_by_edge_class(rid, 'Models')[0]
            neuron_model_dict[self.data.nodes[rid]['name']] = model_rid
        self.rid_dict['NeuronModel'] = neuron_model_dict

        synapse_model_dict = {}
        for synapse_name, rid in self.rid_dict['Synapse'].items():
            model_rid = self.find_predecessor_by_edge_class(rid, 'Models')[0]
            synapse_model_dict[self.data.nodes[model_rid]['name']] = model_rid
        self.rid_dict['SynapseModel'] = synapse_model_dict

    def _subregion_innervation(self):
        self.subregion_arborization = {}
        for name, rid in self.rid_dict['Subregion'].items():
            tmp1 = set(self.find_predecessor_by_edge_class(rid, 'ArborizesIn'))
            self.subregion_arborization[name] = {
                's': set([n for n in tmp1 \
                          for v in self.data.get_edge_data(n, rid).values() \
                          if v['class'] == 'ArborizesIn' and 's' in v['kind']]),
                'b': set([n for n in tmp1 \
                          for v in self.data.get_edge_data(n, rid).values() \
                          if v['class'] == 'ArborizesIn' and 'b' in v['kind']])
                }

        self.subregions = Subregion_State(list(self.rid_dict['Subregion']))
        for name, rid in self.rid_dict['Subregion'].items():
            self.subregions.load_synapses(name,
                                          self.find_synapses_in_subregion(name))

    def initialize_diagram_config(self):
        config = {'inactive': {'neuron': {},
                               'synapse': {},
                               'subregion': {}},
                  'active': {'neuron': {},
                             'synapse': {},
                             'subregion': {}}}

        for neuron_name, rid in self.rid_dict['NeuronModel'].items():
            node_data = deepcopy(self.data.nodes[rid])
            new_node_data = {'name': node_data.pop('class'),
                             'states': {}}
            new_node_data['params'] = node_data
            config['active']['neuron'][neuron_name] = new_node_data
        for synapse_name, rid in self.rid_dict['SynapseModel'].items():
            node_data = deepcopy(self.data.nodes[rid])
            new_node_data = {'name': node_data.pop('class'),
                             'states': {}}
            new_node_data['params'] = node_data
            config['active']['synapse'][synapse_name] = new_node_data
        for subregion_name, rid in self.rid_dict['Subregion'].items():
            config['active']['subregion'][subregion_name] = {}

        self.config = config
        self.send_to_GFX()

    def send_to_GFX(self):
        print('sending circuit configuration to GFX')
        config = {'cx': {k: v for items in self.config['active'].values()\
                            for k, v in items.items()}}
        config['cx']['disabled'] = self.all_inactive()

        config_tosend = json.dumps(config)
        self.fbl.JSCall(messageType = 'setExperimentConfig',
                        data = config_tosend)
        time.sleep(5)

    def colorByUname(self, unames, color):
        _ids = []
        for _u in unames:
            if _u in self.rid_dict['Neuron']:
                _ids.append(self.rid_dict['Neuron'][_u])

        a = {'data': {'commands': {'setcolor': [_ids, color]}},
                                 'messageType': 'Command',
                                 'widget': 'NLP'}
        self.fbl.tryComms(a)

    def initialize_new_model(self, model_version):
        query_list = [
            {"action": {"method":{"query":{"name":self.CX_LPUs, "version":model_version}}},
             "object":{"class":"LPU"}},
            {"action":{"method":{"query":{"name":self.CX_Patterns, "version":model_version}}},
             "object":{"class":"Pattern"}},
            {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
            ]
        data = self.na_query(query_list)
        new_version_nodes = data['nodes']
        if new_version_nodes:
            raise ValueError('Version {} already exist for {}.\
                              Use a different version'.format(
                                model_version,
                                ', '.join(node['name'] for node \
                                          in new_version_nodes.values())),
                            )

        self.model_version = model_version
        self.new_model_initialization = True
        self.updated = {}
        self.added = {}
        self.disabled = {}
        self.add_node_count = itertools.count()

    def create_model(self, model_version):
        if self._is_version_used(model_version):
            raise ValueError('Version {} already exist for {}.\
                              Use a different version'.format(
                                model_version,
                                ', '.join(node['name'] for node \
                                          in new_version_nodes.values())),
                            )
        self.model_version = model_version


    def _is_version_used(self, model_version):
        query_list = [
            {"action": {"method":{"query":{"name":self.CX_LPUs, "version":model_version}}},
             "object":{"class":"LPU"}},
            {"action":{"method":{"query":{"name":self.CX_Patterns, "version":model_version}}},
             "object":{"class":"Pattern"}},
            {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
            ]
        data = self.na_query(query_list)
        new_version_nodes = data['nodes']
        if new_version_nodes:
            return True
        else:
            return False

    ## CAllBACK
    def loadExperimentConfig(self, simExperimentConfig):
        lastObject = self.fbl.simExperimentConfig['lastObject']
        lastLabel = self.fbl.simExperimentConfig['lastLabel']
        action = self.fbl.simExperimentConfig['lastAction']
        self.test.append([lastObject, lastLabel, action])
        print(lastObject, lastLabel, action)
        if lastObject == "neuron":
            if action == "deactivated":
                self._disable_neuron(lastLabel)
            elif action == "activated":
                self._enable_neuron(lastLabel)
            elif action == "toggled":
                if lastLabel in self.config['active']['neuron']:
                    self._disable_neuron(lastLabel)
                elif lastLabel in self.config['inactive']['neuron']:
                    self._enable_neuron(lastLabel)
        elif lastObject == "synapse":
            if lastLabel in self.config['active']['synapse']:
                self._disable_synapse(lastLabel)
            elif lastLabel in self.config['inactive']['synapse']:
                self._enable_synapse(lastLabel)
        elif lastObject == "region":
            if action == "deactivated":
                self._disable_subregion(lastLabel)
            elif action == "activated":
                self._enable_subregion(lastLabel)
        self.send_to_GFX()

    ## Removing Components

    def show_disabled(self):
        print(self.disabled)

    def all_inactive(self):
        return list(self.config['inactive']['neuron'].keys()) + \
               list(self.config['inactive']['synapse'].keys()) + \
               list(self.config['inactive']['subregion'].keys())

    def all_inactive_executable(self):
        return list(self.config['inactive']['neuron'].keys()) + \
               list(self.config['inactive']['synapse'].keys())

    def disable_all(self):
        """
        Remove all neurons and subregions (thereby all synapses) in the diagram.
        """

        all_elements = list(self.rid_dict['Subregion'].keys()) + \
                       list(self.rid_dict['NeuronModel'].keys()) + \
                       list(self.rid_dict['SynapseModel'].keys())

        active_neurons = list(self.config['active']['neuron'])
        for neuron_name in active_neurons:
            self.config['inactive']['neuron'][neuron_name] = \
                self.config['active']['neuron'].pop(neuron_name)
        active_synapses = list(self.config['active']['synapse'])
        for synapse_name in active_synapses:
            self.config['inactive']['synapse'][synapse_name] = \
                self.config['active']['synapse'].pop(synapse_name)
        active_subregions = list(self.config['active']['subregion'])
        for subregion_name in active_subregions:
            self.config['inactive']['subregion'][subregion_name] = \
                self.config['active']['subregion'].pop(subregion_name)
            self.subregions.disable_subregion(subregion_name)

        self.send_to_GFX()

    # @property
    # def disabled(self):
    #     return self.fbl.simExperimentConfig['cx']['disabled']

    @property
    def disabled(self):
        return {'neuron': list(self.config['inactive']['neuron'].keys()),
                'synapse': list(self.config['inactive']['synapse'].keys()),
                'subregion': list(self.config['inactive']['subregion'].keys())}

    def _disable_single_neuron(self, neuron_name):
        """
        Main part of single neuron removal.

        Steps: 1. Remove entries in subregion_arborization
               1. remove edges assocated with ArborizesIn
               2. remove synapses that are assciated with the neuron
               3. remove all edges associated with the neuron
        """
        rid = self.rid_dict['Neuron'][neuron_name]

        in_edges = self.data.in_edges(rid, data = True)
        out_edges = self.data.out_edges(rid, data = True)

        for pre, post, k in list(in_edges):
            if k['class'] == 'SendsTo':
                if self.data.nodes[pre]['class'] == 'Synapse':
                    self._disable_synapse(self.data.nodes[pre]['name'])

        for pre, post, k in list(out_edges):
            if k['class'] == 'SendsTo':
                if self.data.nodes[post]['class'] == 'Synapse':
                    self._disable_synapse(self.data.nodes[post]['name'])

    def _disable_neuron(self, neuron_name):
        """
        Remove a neuron from the diagram.
        Should be called by GUI after simExperimentConfig is updated.

        Parameters
        ----------
        neuron_name: str
                     name of the neuron
        """
        if neuron_name in self.config['active']['neuron']:
            self.config['inactive']['neuron'][neuron_name] = \
                self.config['active']['neuron'].pop(neuron_name)
            self._disable_single_neuron(neuron_name)


    def disable_neurons(self, neurons):
        """
        Disable neurons that exist in the diagram.

        Parameters
        ----------
        neurons:  list of neuron names
        """
        for neuron in neurons:
            self._disable_neuron(neuron)
        self.send_to_GFX()

    def _disable_synapse(self, synapse_name):
        """
        Remove a synapse from the diagram.
        Should be called by GUI after simExperimentConfig is updated.

        Parameters
        ----------
        synapse_name: str
                     name of the synapse
        """
        if synapse_name in self.config['active']['synapse']:
            self.config['inactive']['synapse'][synapse_name] = \
                self.config['active']['synapse'].pop(synapse_name)
            rid = self.rid_dict['Synapse'][synapse_name]
            subregion = self.subregions.disable_synapse(synapse_name)
            if subregion is not None:
                if subregion in self.config['active']['subregion']:
                    self.config['inactive']['subregion'][subregion] = \
                        self.config['active']['subregion'].pop(subregion)
                    # don't remove subregion from self.data

    def disable_synapses(self, synapses):
        """
        Disable synapses that exist in the diagram.

        Parameters
        ----------
        synapses:  list of synapse names
        """
        for synapse in synapses:
            self._disable_synapse(synapse)
        self.send_to_GFX()

    def _disable_subregion(self, subregion_name):
        if subregion_name in self.config['active']['subregion']:
            # self.config['inactive']['subregion'][subregion_name] = \
            #         self.config['active']['subregion'].pop(subregion_name)
            for synapse in list(self.subregions.all_synapses[subregion_name]):
                self._disable_synapse(synapse)

    def disable_subregions(self, subregions):
        """
        Disable synapses that exist in the diagram.

        Parameters
        ----------
        subregions:  list of subregion names
        """
        for subregion in subregions:
            self._disable_subregion(subregion)
        self.send_to_GFX()

    def disable_neuron_family(self, family):
        neuron_names = list(self.neurons_of_family(family).keys())
        self.disable_neurons(neuron_names)

    def disable_neuron_families(self, families):
        neuron_names = sum([list(self.neurons_of_family(family).keys()) for family in families], [])
        self.disable_neurons(neuron_names)

    def remove_components(self):
        """
        Query data base to remove disabled neurons and synapses.
        This should be called right before self.execution()
        """
        disabled_neurons = list(self.config['inactive']['neuron'].keys())
        disabled_synapses = list(self.config['inactive']['synapse'].keys())
        query_list = [
                {"action": {"method":
                    {"query":{"name": self.CX_LPUs,
                              "version": self.initial_model_version}}},
                 "object":{"class":"LPU"}},
                {"action":{"method":
                    {"query":{"name": self.CX_Patterns,
                              "version": self.initial_model_version}}},
                 "object":{"class":"Pattern"}},
                {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                {"action": {"method":{"traverse_owns":{}}},
                 "object":{"memory":0}}
              ]
        self.querier.na_query(query_list, format = 'no_result')

        if len(disabled_neurons):
            query_list = [{"action":{"method":{"has":{"name": disabled_neurons}}},"object":{"state":0}},
                    {"action":{"method":{"get_connected_ports":{}}},"object":{"memory":0}},
                    {"action":{"op":{"find_matching_ports_from_selector":{"memory":0}}},"object":{"state":0}},
                    {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                    {"action":{"method":{"gen_traversal_in":{"min_depth":1, "pass_through":["SendsTo", "SynapseModel","instanceof"]}}},"object":{"memory":0}},
                    {"action":{"method":{"gen_traversal_out":{"min_depth":1, "pass_through":["SendsTo", "SynapseModel","instanceof"]}}},"object":{"memory":1}},
                    {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                    {"action":{"op":{"__add__":{"memory":3}}},"object":{"memory":0}},
                    {"action":{"op":{"__sub__":{"memory":0}}},"object":{"state":0}}]
        else:
            query_list = []

        if len(disabled_synapses):
            if len(query_list):
                query_list.extend(
                    [{"action":{"method":{"has":{"name": disabled_synapses}}},"object":{"state":0}},
                    {"action":{"op":{"__sub__":{"memory":0}}},"object":{"memory":1}}])
            else:
                query_list.extend(
                    [{"action":{"method":{"has":{"name": disabled_synapses}}},"object":{"state":0}},
                    {"action":{"op":{"__sub__":{"memory":0}}},"object":{"state":0}}])

        res = self.querier.na_query(query_list)
        return res

    ## End of Removing Components

    def execute(self, input_processors = {}, output_processors = {},
                steps = None, dt = None, name = 'cx'):
        res_info = self.fbl.client.session.call(u'ffbo.processor.server_information')
        msg = {"user": self.fbl.client._async_session._session_id,
               "servers": {'na': self.fbl.naServerID, 'nk': list(res_info['nk'].keys())[0]}}
        res = self.fbl.client.session.call(u'ffbo.na.query.{}'.format(msg['servers']['na']), {'user': msg['user'],
                                'command': {"retrieve":{"state":0}},
                                'format': "nk"})
        self.fbl.execute_multilpu(name, inputProcessors = input_processors,
                                  outputProcessors = output_processors,
                                  steps = steps, dt = dt)

    def get_result(self, name = 'cx'):
        label_dict = {rid: n for rid, n in self.data.nodes(data='name')}
        self.fbl.updateSimResultLabel(name, label_dict)
        self.result = self.fbl.exec_result[name]
        return self.fbl.exec_result[name]

    def get_result_old(self):
        i = -1
        while True:
            if self.fbl.data[i]['messageType'] != 'Data':
                i -= 1
                continue
            else:
                try:
                    temp = msgpack.unpackb(self.fbl.data[i]['data']['data'])
                except (msgpack.ExtraData, msgpack.UnpackValueError):
                    i -= 1
                    continue
                else:
                    if temp == 'execution_result_end':
                        i -= 1
                        break

        all_data = []
        while self.fbl.data[i]['messageType'] == 'Data':
            try:
                temp = msgpack.unpackb(self.fbl.data[i]['data']['data'])
            except (msgpack.ExtraData, msgpack.UnpackValueError):
                all_data.append(self.fbl.data[i]['data']['data'])
            else:
                if temp == 'execution_result_start':
                    break
                else:
                    all_data.append(temp)
            i -= 1

        temp = msgpack.unpackb(b''.join(all_data[::-1]))
        if 'error' in temp:
            print(temp['error']['message'], file = sys.stderr)
            raise ValueError(temp['error']['exception'])

        meta = temp['success'].pop('meta')
        temp = temp['success']['result']

        result = {'sensory': {}, 'input': {}, 'output': {}, 'meta': meta}
        for data_type, data in temp.items():
            for key, value in data.items():
                k = eval(key).decode('utf-8') if key[0]=='b' else key
                if data_type == 'sensory':
                    result[data_type][k] = [{'dt': val['dt'],
                                             'data': val['data']}#np.array(val['data'])}\
                                            for val in value]
                else:
                    try:
                        node = self.data.nodes[k]
                    except KeyError:
                        pass
                    if node['class'] == 'Port':
                        continue
                    name = node['name']
                    result[data_type][name] = {k: {'data': v['data'],# np.array(v['data']),
                                                   'dt': v['dt']} \
                                               for k, v in value.items()}
        self.result = result
        return result

    def enable_neuron_family(self, family):
        neuron_names = list(self.neurons_of_family(family, active = False).keys())
        self.enable_neurons(neuron_names)

    def enable_neurons(self, neurons):
        """
        Enable neurons that exist in the diagram.

        Parameters
        ----------
        neurons:  list of neuron names
        """
        for neuron in neurons:
            self._enable_neuron(neuron)
        self.send_to_GFX()

    def _enable_neuron(self, neuron_name):
        """
        Add a neuron that exists in diagram.
        Should be called by GUI after simExperimentConfig is updated.

        Parameters
        ----------
        neuron_name: str
                     name of the neuron
        """
        if neuron_name in self.config['inactive']['neuron']:
            self.config['active']['neuron'][neuron_name] = \
                self.config['inactive']['neuron'].pop(neuron_name)

            rid = self.rid_dict['Neuron'][neuron_name]

            in_edges = self.data.in_edges(rid, data = True)
            out_edges = self.data.out_edges(rid, data = True)

            for pre, post, k in list(in_edges):
                if k['class'] == 'SendsTo':
                    if self.data.nodes[pre]['class'] == 'Synapse':
                        self._enable_synapse(self.data.nodes[pre]['name'])

            for pre, post, k in list(out_edges):
                if k['class'] == 'SendsTo':
                    if self.data.nodes[post]['class'] == 'Synapse':
                        self._enable_synapse(self.data.nodes[post]['name'])

    def enable_synapses(self, synapses):
        """
        Enable synapses that exist in the diagram.

        Parameters
        ----------
        synapses:  list of synapse names
        """
        for synapse in synapses:
            self._enable_synapse(synapse)
        self.send_to_GFX()

    def _enable_synapse(self, synapse_name):
        if synapse_name in self.config['inactive']['synapse']:
            self.config['active']['synapse'][synapse_name] = \
                self.config['inactive']['synapse'].pop(synapse_name)
            rid = self.rid_dict['Synapse'][synapse_name]
            subregion = self.subregions.enable_synapse(synapse_name)
            if subregion is not None:
                if subregion in self.config['inactive']['subregion']:
                    self.config['active']['subregion'][subregion] = \
                        self.config['inactive']['subregion'].pop(subregion)

    def enable_subregions(self, subregions):
        """
        Enable subregions that exist in the diagram.

        Parameters
        ----------
        subregions:  list of subregion names
        """
        for subregion in subregions:
            self._enable_subregion(subregion)
        self.send_to_GFX()

    def _enable_subregion(self, subregion_name):
        if subregion_name in self.config['inactive']['subregion']:
            for synapse in list(self.subregions.disabled_synapses[subregion_name]):
                self._enable_synapse(synapse)

    def process_configuration(self):
        updated = self.fbl.simExperimentConfig['cx']['updated']
        disabled = self.fbl.simExperimentConfig['cx']['disabled']
        added = self.fbl.simExperimentConfig['cx']['added']


    ## Queries on the Graph

    def owned_by(self, rid, cls):
        owners = self.find_predecessor_by_edge_class(rid, 'Owns')
        pre = [d for d in owners if self.data.nodes[d]['class'] == cls]
        return pre

    def owns(self, rid, cls):
        tmp = self.find_successor_by_edge_class(rid, 'Owns')
        post = [d for d in tmp if self.data.nodes[d]['class'] == cls]
        return post

    def find_predecessor_by_edge_class(self, rid, edge_class, edge_value = None):
        if edge_value is None:
            return [m for m, n, c in self.data.in_edges(rid, data = 'class') \
                    if c == edge_class]
        else:
            return [m for m, n, c in self.data.in_edges(rid) \
                    if c == edge_class and \
                    all([k in c and c[k] == v for k, v in edge_values.items()])]

    def find_successor_by_edge_class(self, rid, edge_class, edge_value = None):
        if edge_value is None:
            return [n for m, n, c in self.data.out_edges(rid, data = 'class') \
                    if c == edge_class]
        else:
            return [n for m, n, c in self.data.out_edges(rid) \
                    if c == edge_class and \
                    all([k in c and c[k] == v for k, v in edge_values.items()])]

    def find_synapses_in_subregion(self, subregion_name):
        rids =  self.owns(self.rid_dict['Subregion'][subregion_name], 'Synapse')
        synapses = [self.data.nodes[k]['name'] for k in rids]
        return synapses

    def find_models(self, rid):
        return self.find_predecessor_by_edge_class(rid, 'Models')

    def check_if_neuron_exists(self, neuron_name):
        """
        Check if a neuron exists that arborizes in the same regions as
        the neuron_name.
        If it exists, return the rid of the neuron.
        If it does not exist, return None
        """
        parsed = self.parser.parse(neuron_name)
        list_of_neurons = []
        try:
            count = 0
            for item in parsed:
                for region in item['regions']:
                    for neurite in item['neurite']:
                        rids = self.subregion_arborization['{}/{}'.format(
                                        item['neuropil'], region)][neurite]
                        if count == 0:
                            common = rids
                        else:
                            common = common.intersection(rids)
                            assert(len(common))
                        count += 1
        except AssertionError:
            return None

        exact = []
        for n in common:
            parsed_n = self.parser.parse(self.data.nodes[n]['name'])
            if len(parsed_n) != len(parsed):
                continue
            else:
                if parsed_n == parsed:
                    exact.append(n)
        print(exact)
        if len(exact) == 0:
            return None
        else:
            assert(len(exact) == 1)
            return exact[0]

    ## Operations on the Graph
    def add_SendsTo(self, a, b):
        self.data.add_edge(a, b, **{'class': 'SendsTo'})

    def add_Owns(self, a, b):
        self.data.add_edge(a, b, **{'class': 'Owns'})

    def add_Models(self, a, b, version):
        self.data.add_edge(a, b, **{'class': 'Models', 'version': version})

    def add_node(self, rid, attr_dict):
        self.data.add_node(rid, **attr_dict)

    def add_ArborizesIn(self, neuron_rid, subregion_rid, neurite_type):
        self.data.add_edge(neuron_rid, subregion_rid,
                           **{'class': 'ArborizesIn',
                              'kind': set(neurite_type)
                             })
        subregion_name = self.data.nodes[subregion_rid]['name']
        for k in neurite_type:
            self.subregion_arborization[subregion_name][k].add(neuron_rid)

    def add_Tract(self, neuropil_pair):
        tract_name = '-'.join(sorted(list(neuropil_pair)))
        rid = self.get_next_rid()
        self.add_node(rid,
                      {"name": tract_name,
                       "version": self.model_version,
                       "neuropils": set(neuropil_pair)})
        return rid

    ## End of Operations on the Graph
    def get_node_name(self, rid):
        return self.data.nodes[rid]['name']

    def get_subregion_name(self, region, neuropil):
        if isinstance(region, tuple):
            subregion_name = '{}/({})'.format(neuropil, ','.join(region))
        else:
            subregion_name = '{}/{}'.format(neuropil, region)
        return subregion_name

    def get_next_rid(self):
        return '{}'.format(next(self.add_node_count))

    def neurons_of_family(self, family, active = True):
        """
        Returns a dictory of names/rid pairs of active neuron that
        belongs to the family

        Parameters
        ----------
        family: a Neuron family name (e.g., PB-EB-LAL)
        """
        if active:
            return {neuron_name: rid for neuron_name, rid in self.rid_dict['Neuron'].items()\
                    if self.data.nodes[rid]['family'] == family and \
                    neuron_name in self.config['active']['neuron']}
        else:
            return {neuron_name: rid for neuron_name, rid in self.rid_dict['Neuron'].items()\
                    if self.data.nodes[rid]['family'] == family}


    def neuron_uid_by_family(self, family, subregion, neurite = 's'):
        """
        Returns the uid of neuron that belongs to the family
        and that has dendrites innervating subregion

        Parameters
        ----------
        family: a Neuron family name (e.g., PB-EB-LAL)
        subregion: name of the subregion that the neuron has dendrites
        """
        rids = set([rid for neuron_name, rid in self.rid_dict['Neuron'].items()\
                if self.data.nodes[rid]['family'] == family and \
                neuron_name in self.config['active']['neuron']])
        neurons = []
        for rid in rids.intersection(self.subregion_arborization[subregion][neurite]):
            neurons.extend(self.find_predecessor_by_edge_class(rid, 'Models'))
        return neurons

    @classmethod
    def raster_plot(cls, data, max_time = np.inf, xlim = None):
        fig1, axes = plt.subplots(1,1, figsize=(10,8))
        fig1.subplots_adjust(left = 0.3)
        j = 0
        for name, spike_time in data:
            axes.eventplot(np.select([spike_time < max_time], [spike_time]),
                           lineoffsets = j, linelengths = 0.5)
            j += 1
        plt.grid(b=True, which = 'minor', axis='y')
        axes.set_xlim(left = 0, right = xlim)
        axes.set_yticks(range(j))
        axes.set_yticklabels([name for name, spike_time in data])
        plt.show()

    @classmethod
    def _raster_plot(cls, axes, data, max_time = np.inf, xlim = None):
        j = 0
        for name, spike_time in data:
            axes.eventplot(np.select([spike_time < max_time], [spike_time]),
                           lineoffsets = j, linelengths = 0.5)
            j += 1

    def plot_PB_EB_LAL(self, order = 'PB'):
        data = self.get_PB_EB_LAL_data(order = order)
        self.raster_plot(data)

    def get_PB_EB_LAL_data(self, order = 'PB'):
        if order == 'PB':
            uids = [self.neuron_uid_by_family('PB-EB-LAL', 'PB/L{}'.format(10-i)) \
                    for i in range(1,10)]
            uids.extend([self.neuron_uid_by_family('PB-EB-LAL', 'PB/R{}'.format(i)) \
                         for i in range(1,10)])
        elif order == 'EB':
            uids = [self.neuron_uid_by_family('PB-EB-LAL', 'EB/L{}'.format(9-i), 'b') \
                    for i in range(1,9)]
            uids.extend([self.neuron_uid_by_family('PB-EB-LAL', 'EB/R{}'.format(i), 'b') \
                         for i in range(1,9)])
        else:
            raise ValueError('Order has to be either PB or EB.')
        data = []
        outputs = self.result['output']
        for uid in uids:
            if len(uid):
                names = [self.data.nodes[n]['name'] for n in uid]
                for name in names:
                    if name in outputs:
                        spike_time = outputs[name]['spike_time']['data']
                        data.append((name, spike_time))
        return data

    def get_PB_FB_LAL_data(self, order = 'PB'):
        if order == 'PB':
            uids = [self.neuron_uid_by_family('PB-FB-LAL', 'PB/L{}'.format(10-i)) \
                    for i in range(1,10)]
            uids.extend([self.neuron_uid_by_family('PB-FB-LAL', 'PB/R{}'.format(i)) \
                         for i in range(1,10)])

        data = []
        outputs = self.result['output']
        for uid in uids:
            if len(uid):
                names = [self.data.nodes[n]['name'] for n in uid]
                for name in names:
                    if name in outputs:
                        spike_time = outputs[name]['spike_time']['data']
                        data.append((name, spike_time))
        return data

    def get_PB_Local_data(self):
        a = sorted(self.neurons_of_family('PB').items())
        b = a[0:5][::-1]+a[5:]
        uids = [n[1] for n in b]

        data = []
        outputs = self.result['output']
        for uid in uids:
            name = self.data.nodes[uid]['name']
            if name in outputs:
                spike_time = outputs[name]['spike_time']['data']
                data.append((name, spike_time))
        return data

    def get_PB_FB_NO_data(self, order = 'PB'):
        if order == 'PB':
            uids = [self.neuron_uid_by_family('PB-FB-NO', 'PB/L{}'.format(10-i)) \
                    for i in range(1,10)]
            uids.extend([self.neuron_uid_by_family('PB-FB-NO', 'PB/R{}'.format(i)) \
                         for i in range(1,10)])

        data = []
        outputs = self.result['output']
        for uid in uids:
            if len(uid):
                names = [self.data.nodes[n]['name'] for n in uid]
                for name in names:
                    if name in outputs:
                        spike_time = outputs[name]['spike_time']['data']
                        data.append((name, spike_time))
        return data

    def get_PB_EB_NO_data(self, order = 'PB'):
        if order == 'PB':
            uids = [self.neuron_uid_by_family('PB-EB-NO', 'PB/L{}'.format(10-i)) \
                    for i in range(1,10)]
            uids.extend([self.neuron_uid_by_family('PB-EB-NO', 'PB/R{}'.format(i)) \
                         for i in range(1,10)])
        elif order == 'EB':
            uids = [self.neuron_uid_by_family('PB-EB-NO', 'EB/L{}'.format(9-i), 'b') \
                    for i in range(1,9)]
            uids.extend([self.neuron_uid_by_family('PB-EB-NO', 'EB/R{}'.format(i), 'b') \
                         for i in range(1,9)])
        else:
            raise ValueError('Order has to be either PB or EB.')
        data = []
        outputs = self.result['output']
        for uid in uids:
            if len(uid):
                names = [self.data.nodes[n]['name'] for n in uid]
                for name in names:
                    if name in outputs:
                        spike_time = outputs[name]['spike_time']['data']
                        data.append((name, spike_time))
        return data

    def get_EB_LAL_PB_data(self, order = 'PB'):
        if order == 'PB':
            uids = [self.neuron_uid_by_family('EB-LAL-PB', 'PB/L{}'.format(10-i), 'b') \
                    for i in range(1,10)]
            uids.extend([self.neuron_uid_by_family('EB-LAL-PB', 'PB/R{}'.format(i), 'b') \
                         for i in range(1,10)])
        elif order == 'EB':
            uids = [self.neuron_uid_by_family('EB-LAL-PB', 'EB/L{}'.format(9-i), 'b') \
                    for i in range(1,9)]
            uids.extend([self.neuron_uid_by_family('EB-LAL-PB', 'EB/R{}'.format(i), 'b') \
                         for i in range(1,9)])
        else:
            raise ValueError('Order has to be either PB or EB.')
        data = []
        outputs = self.result['output']
        for uid in uids:
            if len(uid):
                names = [self.data.nodes[n]['name'] for n in uid]
                for name in names:
                    if name in outputs:
                        spike_time = outputs[name]['spike_time']['data']
                        data.append((name, spike_time))
        return data

    def animate_PB_EB_LAL(self, order = 'PB', sensory_data = 'BU'):
        """
        Generate an animation object that can be played back
        in a notebook by HTML(anim.to_html5_video())
        """
        data = self.get_PB_EB_LAL_data(order = order)
        video = self.result['sensory'][sensory_data][0]['data']
        dt = self.result['sensory'][sensory_data][0]['dt']
        dur = self.result['meta']['dur']

        fig, ax = plt.subplots(2, 1, gridspec_kw = {'height_ratios': [1,3]},
                               figsize = (10,8))
        fig.subplots_adjust(left = 0.3)
        ax[0].set_axis_off()
        ax[1].set_xlim(right = dur)
        ax[1].set_yticks(range(len(data)))
        ax[1].set_yticklabels([name for name, _ in data])
        max_time = np.inf

        def animate(i):
            ax[0].imshow(video[i])
            j = 0
            for name, spike_time in data:
                if len(spike_time):
                    ax[1].eventplot(np.select([spike_time < dt*i], [spike_time]),
                                    lineoffsets = j, linelength = 0.5)
                j += 1

        anim = animation.FuncAnimation(fig, animate, frames = video.shape[0],
                                       interval = dur*10/video.shape[0]*1000)
        return anim

    def animate_PB_EB_NO(self, order = 'PB', sensory_data = 'BU'):
        """
        Generate an animation object that can be played back
        in a notebook by HTML(anim.to_html5_video())
        """
        data = self.get_PB_EB_NO_data(order = order)
        video = self.result['sensory'][sensory_data][0]['data']
        dt = self.result['sensory'][sensory_data][0]['dt']
        dur = self.result['meta']['dur']

        fig, ax = plt.subplots(2, 1, gridspec_kw = {'height_ratios': [1,3]},
                               figsize = (10,8))
        fig.subplots_adjust(left = 0.3)
        ax[0].set_axis_off()
        ax[1].set_xlim(right = dur)
        ax[1].set_yticks(range(len(data)))
        ax[1].set_yticklabels([name for name, _ in data])
        max_time = np.inf

        def animate(i):
            ax[0].imshow(video[i])
            j = 0
            for name, spike_time in data:
                if len(spike_time):
                    ax[1].eventplot(np.select([spike_time < dt*i], [spike_time]),
                                    lineoffsets = j, linelength = 0.5)
                j += 1

        anim = animation.FuncAnimation(fig, animate, frames = video.shape[0],
                                       interval = dur*10/video.shape[0]*1000)
        return anim

    def animate_EB_LAL_PB(self, order = 'PB', sensory_data = 'BU'):
        """
        Generate an animation object that can be played back
        in a notebook by HTML(anim.to_html5_video())
        """
        data = self.get_EB_LAL_PB_data(order = order)
        video = self.result['sensory'][sensory_data][0]['data']
        dt = self.result['sensory'][sensory_data][0]['dt']
        dur = self.result['meta']['dur']

        fig, ax = plt.subplots(2, 1, gridspec_kw = {'height_ratios': [1,3]},
                               figsize = (10,8))
        fig.subplots_adjust(left = 0.3)
        ax[0].set_axis_off()
        ax[1].set_xlim(right = dur)
        ax[1].set_yticks(range(len(data)))
        ax[1].set_yticklabels([name for name, _ in data])
        max_time = np.inf

        def animate(i):
            ax[0].imshow(video[i])
            j = 0
            for name, spike_time in data:
                if len(spike_time):
                    ax[1].eventplot(np.select([spike_time < dt*i], [spike_time]),
                                    lineoffsets = j, linelength = 0.5)
                j += 1

        anim = animation.FuncAnimation(fig, animate, frames = video.shape[0],
                                       interval = dur*10/video.shape[0]*1000)
        return anim

    @classmethod
    def psth(cls, data, dur, max_time = np.inf, xlim = None):
        fig1, axes = plt.subplots(1,1, figsize=(10,8))
        fig1.subplots_adjust(left = 0.3)
        j = 0
        for name, spike_time in data:
            rates, stamps = compute_psth(spike_time, 1e-3, 0.05, 1e-3, dur)
            axes.plot(stamps, rates)
            # j += 1
        plt.grid(b=True, which = 'minor', axis='y')
        axes.set_xlim(right = xlim)
        # axes.set_yticks(range(j))
        # axes.set_yticklabels([name for name, spike_time in data])
        plt.show()

    def show_EB_response(self, dur, dt, dpi = 100, figsize=(5,10), vmax = None):
        all_EPGs = self.get_EB_LAL_PB_data(order = 'PB')
        wedges = ['L{}'.format(9-i) for i in range(1,9)] + ['R{}'.format(i) for i in range(1,9)]
        wedge_spikes = {wedge: np.zeros(int(dur/dt)) for wedge in wedges}
        wedge_count = Counter()


        for epg, ss in all_EPGs:
            spikes = np.zeros(int(dur/dt))
            spikes[np.round(ss/dt).astype(np.int32)] = 1
            for wedge in set(sum([list(n['regions']) for n in self.parser.parse(epg) if n['neuropil']=='EB'], [])):
                wedge_spikes[wedge] += spikes
                wedge_count[wedge] += 1

        t = np.arange(0, 5, dt)
        f = np.fft.fft(np.exp(-t/0.7215), n = wedge_spikes['L8'].shape[0]+t.shape[0]-1)

        wedge_spike_rate = np.zeros((wedge_spikes['L8'].shape[0],len(wedges)))

        for i, wedge in enumerate(wedges):
            rate = np.real(np.fft.ifft(np.fft.fft(wedge_spikes[wedge], n = wedge_spikes['L8'].shape[0]+t.shape[0]-1)*f))
            rate = rate[:wedge_spikes['L8'].shape[0]]
            wedge_spike_rate[:,i] = rate/wedge_count[wedge]

        fig1, axes = plt.subplots(1,1, figsize = figsize, dpi=dpi)
        plt.imshow(wedge_spike_rate[:,::-1], aspect =16/(dur/dt)*3, cmap = cm.hot, vmax = vmax);
        axes.set_xlabel('')
        axes.set_ylabel('time (s)', fontsize=20)
        plt.xticks([0,4,7,8,11,15], labels=[wedges[i] for i in [0,4,7,8,11,15]][::-1], fontsize = 8)
        axes.xaxis.set_tick_params(labeltop='on', labelbottom=False, bottom = False)
        axes.set_yticks(np.arange(0,360000, 50000))
        axes.set_yticklabels(np.arange(0,36,5), fontsize = 20)
        plt.grid(False)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20);
        plt.show()
        return wedge_spike_rate

    def show_EB_response_at_step(self, wedge_spike_rate, step, vmax = None):
        fig1, axes = plt.subplots(1,1, figsize=(5,5), dpi=100)
        wedges = ['L{}'.format(9-i) for i in range(1,9)] + ['R{}'.format(i) for i in range(1,9)]
        r = np.arange(4,10)
        theta = -np.arange(-180,180+22,22.5)/180*np.pi+np.pi/2
        R, Theta = np.meshgrid(r, theta)

        X1 = R*np.cos(Theta)
        X2 = R*np.sin(Theta)

        text_r = np.zeros(16)+10
        text_theta = (270-11.25-22.5*np.arange(16))/180*np.pi

        text_x = text_r*np.cos(text_theta)
        text_y = text_r*np.sin(text_theta)

        plt.pcolormesh(X1,X2,np.tile(wedge_spike_rate[step,::-1].reshape(-1,1), [1,6]),
                       cmap=cm.hot, vmin = 0, vmax = vmax, shading = 'auto')
        plt.axis('equal')

        for i, (x,y) in enumerate(zip(text_x, text_y)):
            plt.text(x-1,y-0.75, wedges[15-i], rotation=-11.25-i*22.5 if y < 0 else -11.25-i*22.5+180, fontsize=20)
        plt.text(-1.7, -1, 'EB', fontsize=40)
        plt.axis('off')
        plt.show()

def compute_psth(spike_times, d_t, window, interval, dur):
    """
    Compute the peri-stimulus time histogram.
    Arguments:
        spikes (ndarray): spike sequences.
        d_t (float): time step.
        window (float): the size of the window.
        interval (float): the time shift between two consecutive windows.
    Returns:
        rates (ndarray): the average spike rate for each windows.
        stamps (ndarray): the time stamp for each windows.
    """

    spikes = np.zeros(int(dur/d_t))
    for s in spike_times:
        spikes[int(s/d_t)] += 1

    cum_spikes = np.cumsum(spikes)
    start = np.arange(0., d_t*len(spikes)-window, interval) // d_t
    stop = np.arange(window, d_t*len(spikes)-d_t, interval) // d_t
    start = start.astype(int, copy=False)
    stop = stop.astype(int, copy=False)

    start = start[:len(stop)]

    rates = (cum_spikes[stop] - cum_spikes[start]) / window
    stamps = np.arange(0, len(rates)*interval, interval)

    return rates, stamps
