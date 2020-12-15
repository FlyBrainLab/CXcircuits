


import h5py
import numpy as np

from tqdm import tqdm

from neurokernel.LPU.InputProcessors.BaseInputProcessor import BaseInputProcessor
from neurokernel.LPU.InputProcessors.PresynapticInputProcessor import PresynapticInputProcessor
from .parse_arborization import NeuronArborizationParser


class BU_InputProcessor(BaseInputProcessor):
    def __init__(self, shape, dt, dur, name, video_config, rf_config, neurons,
                 scale = 1.0,
                 record_file = None, record_interval = 1):
        video_cls = Video_factory(video_config.get('type', 'moving_bar_l2r'))
        self.video = video_cls(shape, dt, dur, video_config.get('bar_width', 50),
                               start = video_config.get('start', None),
                               stop = video_config.get('stop', None),
                               record_file = video_config.get('record', None),
                               record_interval = video_config.get('record_interval', 1))
        uids = list(neurons.keys())
        neuron_names = [neurons[n]['name'] for n in uids]
        neuron_ids = np.array([int(name.split('/')[1][1:]) for name in neuron_names])
        neuron_side = set([name.split('/')[1][0] for name in neuron_names])
        if len(neuron_side) > 1:
            raise ValueError('BU neurons must be on one side')
        else:
            self.hemisphere = neuron_side.pop()

        self.fc = CircularGaussianFilterBank(
                        (shape[0], shape[1]),
                        rf_config.get('sigma', 0.05), 10,
                        hemisphere = self.hemisphere)

        self.index = neuron_ids - 1
        var_list = [('I', uids)]
        self.name = name
        self.scale = scale
        # self.n_inputs = 80
        #self.filter_filename = '{}_filters.h5'.format(self.name)
        super(BU_InputProcessor, self).__init__(var_list,
                                                sensory_file = self.video.record_file,
                                                sensory_interval = self.video.record_interval,
                                                input_file = record_file,
                                                input_interval = record_interval)

    def pre_run(self):
        self.video.pre_run()
        self.fc.create_filters()
        # self.file = h5py.File('{}_inputs.h5'.format(self.name), 'w')
        # self.file.create_dataset('I',
        #                          (0, self.n_inputs),
        #                          dtype = np.double,
        #                          maxshape=(None, self.n_inputs))

    def is_input_available(self):
        return True

    def update_input(self):
        frame = self.video.run_step()
        BU_input = self.fc.apply_filters(frame, scale = self.scale).reshape(-1)
        self.variables['I']['input'] = BU_input[self.index]
        # self.record_frame(BU_input)

    # def record_frame(self, input):
    #     self.file['I'].resize((self.file['I'].shape[0]+1, self.n_inputs))
    #     self.file['I'][-1,:] = input

    def __del__(self):
        try:
            self.close_file()
        except:
            pass

class PB_InputProcessor(BaseInputProcessor):
    def __init__(self, shape, dt, dur, name, video_config, rf_config, neurons,
                 scale = 1.0,
                 record_file = None, record_interval = 1):
        video_cls = Video_factory(video_config.get('type', 'moving_bar_l2r'))
        self.video = video_cls(shape, dt, dur, video_config.get('bar_width', 50),
                               start = video_config.get('start', None),
                               stop = video_config.get('stop', None),
                               record_file = video_config.get('record', None),
                               record_interval = video_config.get('record_interval', 1))
        num_glomeruli = rf_config.get('num_glomeruli', 8)
        self.fr = RectangularFilterBank(shape, num_glomeruli)
        self.scale = scale
        uids = list(neurons.keys())

        parser = NeuronArborizationParser()
        new_uids = []
        neuron_ids = []
        for n in uids:
            neuron_name = neurons[n]['name']
            subregions = [u['regions'] for u in parser.parse(neuron_name) if u['neuropil'] == 'PB' and 's' in u['neurite']][0]
            for region in subregions:
                new_uids.append(n)
                if int(region[1:]) == 1:
                    neuron_ids.append(1)
                else:
                    neuron_ids.append( (num_glomeruli + 2 - int(region[1:])) \
                                       if region[0] == 'L' else \
                                       int(region[1:]))

        self.index = np.array(neuron_ids, np.int32) - 1
        var_list = [('I', new_uids)]
        self.name = name
        # self.n_inputs = 18
        super(PB_InputProcessor, self).__init__(var_list,
                                                sensory_file = self.video.record_file,
                                                sensory_interval = self.video.record_interval,
                                                input_file = record_file,
                                                input_interval = record_interval)

    def pre_run(self):
        self.video.pre_run()
        self.fr.create_filters()
        # self.file = h5py.File('{}_inputs.h5'.format(self.name), 'w')
        # self.file.create_dataset('I',
        #                          (0, self.n_inputs),
        #                          dtype = np.double,
        #                          maxshape=(None, self.n_inputs))

    def is_input_available(self):
        return True

    def update_input(self):
        frame = self.video.run_step()
        PB_input = self.fr.apply_filters(frame, scale = self.scale)

        self.variables['I']['input'] = PB_input[self.index]

    def __del__(self):
        try:
            self.close_file()
        except:
            pass


class PB_InputProcessorPaper(BaseInputProcessor):
    def __init__(self, shape, dt, dur, name, video_config, rf_config, neurons,
                 scale = 1.0,
                 record_file = None, record_interval = 1):
        video_cls = Video_factory(video_config.get('type', 'moving_bar_l2r'))
        self.video = video_cls(shape, dt, dur, video_config.get('bar_width', 50),
                               start = video_config.get('start', None),
                               stop = video_config.get('stop', None),
                               record_file = video_config.get('record', None),
                               record_interval = video_config.get('record_interval', 1))
        num_glomeruli = rf_config.get('num_glomeruli', 18)
        self.fr = RectangularFilterBank(shape, num_glomeruli)
        self.scale = scale
        uids = list(neurons.keys())

        parser = NeuronArborizationParser()
        new_uids = []
        neuron_ids = []
        for n in uids:
            neuron_name = neurons[n]['name']
            subregions = [u['regions'] for u in parser.parse(neuron_name) if u['neuropil'] == 'PB' and 's' in u['neurite']][0]
            for region in subregions:
                new_uids.append(n)
                neuron_ids.append(
                    (num_glomeruli//2+1 - int(region[1:])) \
                     if region[0] == 'L' else\
                     int(region[1:]) + num_glomeruli//2)

        self.index = np.array(neuron_ids, np.int32) - 1
        var_list = [('I', new_uids)]
        self.name = name
        # self.n_inputs = 18
        super(PB_InputProcessor, self).__init__(var_list,
                                                sensory_file = self.video.record_file,
                                                sensory_interval = self.video.record_interval,
                                                input_file = record_file,
                                                input_interval = record_interval)

    def pre_run(self):
        self.video.pre_run()
        self.fr.create_filters()

    def is_input_available(self):
        return True

    def update_input(self):
        frame = self.video.run_step()
        PB_input = self.fr.apply_filters(frame, scale = self.scale)

        self.variables['I']['input'] = PB_input[self.index]

    def __del__(self):
        try:
            self.close_file()
        except:
            pass


class EB_InputProcessor(BaseInputProcessor):
    def __init__(self, shape, dt, dur, name, video_config, rf_config, neurons,
                 scale = 1.0,
                 record_file = None, record_interval = 1):
        video_cls = Video_factory(video_config.get('type', 'moving_bar_l2r'))
        self.video = video_cls(shape, dt, dur, video_config.get('bar_width', 50),
                               start = video_config.get('start', None),
                               stop = video_config.get('stop', None),
                               record_file = video_config.get('record', None),
                               record_interval = video_config.get('record_interval', 1))
        num_glomeruli = rf_config.get('num_glomeruli', 16)
        self.fr = RectangularFilterBank(shape, num_glomeruli)
        self.scale = scale
        uids = list(neurons.keys())

        parser = NeuronArborizationParser()
        new_uids = []
        neuron_ids = []
        for n in uids:
            neuron_name = neurons[n]['name']
            subregions = set()
            for u in parser.parse(neuron_name):
                if u['neuropil'] == 'EB' and 's' in u['neurite']:
                     subregions |= u['regions']
            for region in subregions:
                new_uids.append(n)
                neuron_ids.append(
                    (num_glomeruli//2+1 - int(region[1:])) \
                     if region[0] == 'L' else\
                     int(region[1:]) + num_glomeruli//2)

        self.index = np.array(neuron_ids, dtype = np.int32) - 1
        var_list = [('I', new_uids)]
        self.name = name
        # self.n_inputs = 18
        super(EB_InputProcessor, self).__init__(var_list,
                                                sensory_file = self.video.record_file,
                                                sensory_interval = self.video.record_interval,
                                                input_file = record_file,
                                                input_interval = record_interval)

    def pre_run(self):
        self.video.pre_run()
        self.fr.create_filters()

    def is_input_available(self):
        return True

    def update_input(self):
        frame = self.video.run_step()
        EB_input = self.fr.apply_filters(frame, scale = self.scale)

        self.variables['I']['input'] = EB_input[self.index]
        # self.record_frame(PB_input)

    def __del__(self):
        try:
            self.close_file()
        except:
            pass


class EB_Kakaria_InputProcessor(PresynapticInputProcessor):
    def __init__(self, shape, dt, dur, name, video_config, rf_config, neurons,
                 scale = 1.0,
                 record_file = None, record_interval = 1):
        uids = list(neurons.keys())

        parser = NeuronArborizationParser()
        new_uids = []
        neuron_ids = []
        wedge_map = {'L{}'.format(i): (0+22.5*(i-1), 22.5*i) for i in range(1, 9)}
        wedge_map.update({'R{}'.format(i): (-22.5*i, -22.5*(i-1)) for i in range(1, 9)})
        input_mapping = {}
        for n in uids:
            neuron_name = neurons[n]['name']
            subregions = set()
            for u in parser.parse(neuron_name):
                if u['neuropil'] == 'EB' and 's' in u['neurite']:
                     subregions |= u['regions']
            if len(subregions) == 2:
                new_uids.append(n)
                input_mapping[n] = []
                for region in subregions:
                    input_mapping[n].append(wedge_map[region])

        steps = int(dur/dt)

        # Define Inputs
        Gposes = []
        Gweights = []

        Gpos = np.zeros(steps)
        Gweight = np.zeros(steps)

        t = np.arange(0, dur, dt)
        Gweight[int(np.round(0.0/dt)):int(np.round(30.0/dt))] = 1.0
        Gpos[int(np.round(0.0/dt)):int(np.round(1.0/dt))] = -180.
        Gpos[int(np.round(1.0/dt)):int(np.round(17.0/dt))] = -180+22.5*np.arange(0,16, dt)
        Gpos[int(np.round(17.0/dt)):int(np.round(30.0/dt))] = 180-22.5*np.arange(0,13, dt)
        # Gpos[int(np.round(30.0/dt)):int(np.round(31.0/dt))] = -180+22.5*2
        Gposes.append(Gpos)
        Gweights.append(Gweight)

        Gpos = np.zeros(steps)
        Gweight = np.zeros(steps)
        Gweight[int(np.round(0.0/dt)):int(np.round(33.0/dt))] = 0.4
        Gpos[int(np.round(0.0/dt)):int(np.round(33.0/dt))] = 60.

        Gposes.append(Gpos)
        Gweights.append(Gweight)

        x = np.arange(-180,181)
        r = np.zeros((steps, len(input_mapping)))
        inputs = np.empty((steps, len(x)))

        for Gpos, Gweight in zip(Gposes, Gweights):
            for i in tqdm(range(steps)):
                inputs[i,:] = np.exp(50*np.cos((x-Gpos[i])/180*np.pi))/1.842577884719606e+21
            for j, uid in enumerate(new_uids):
                v = np.zeros(steps)
                for start,end in input_mapping[uid]:
                    v += inputs[:, int(np.round(start))+180:int(np.round(end))+180].sum(axis=1)/180*np.pi
                r[:,j] += v*Gweight
        r = np.minimum(r,1.)*120+5
        del inputs

        times = []
        index = []
        for j, uid in enumerate(new_uids):
            spike_times = np.where(np.random.rand(steps) < r[:,j]*dt)[0]*dt
            times.append(np.array(spike_times))
            index.append(np.zeros(len(spike_times), dtype = np.int32)+j)

        all_times = np.concatenate(times)
        all_index = np.concatenate(index)
        idx = np.argsort(all_times)
        all_times = all_times[idx]
        all_index = all_index[idx]
        data = {'time': all_times, 'index': all_index}

        super(EB_Kakaria_InputProcessor, self).__init__(
                    {'spike_state': {'uids': new_uids, 'data': data,
                                     'components': {'class': 'AlphaSynapse',
                                                    'gmax': 0.0175*0.8,
                                                    'ar': 0.9,
                                                    'ad': 0.15,
                                                    'reverse': 0.}
                                    },
                    },
                    input_file = None)#)record_file,
                    # input_interval = record_interval)
                    #sensory_file = self.video.record_file,
                    #sensory_interval = self.video.record_interval,)
        self.to_file(record_file)


class PB_Su_Bilateral_InputProcessor(PresynapticInputProcessor):
    def __init__(self, shape, dt, dur, name, video_config, rf_config, neurons,
                 scale = 1.0,
                 record_file = None, record_interval = 1):
        """
        neurons: names of PB-EB-LAL neurons in {uid: name ... } dict.
        """
        PB_bilateral_input_rate = 50.
        uids = list(neurons.keys())

        parser = NeuronArborizationParser()
        new_uids = []
        neuron_ids = []
        wedge_map = {'L{}'.format(i): (0+22.5*(i-1), 22.5*i) for i in range(1, 9)}
        wedge_map.update({'R{}'.format(i): (-22.5*i, -22.5*(i-1)) for i in range(1, 9)})
        input_mapping = {}
        for n in uids:
            neuron_name = neurons[n]['name']
            subregions = set()
            for u in parser.parse(neuron_name):
                if u['neuropil'] == 'EB' and 'b' in u['neurite']:
                     subregions |= u['regions']
            new_uids.append(n)
            input_mapping[n] = []
            for region in subregions:
                input_mapping[n].append(wedge_map[region])

        steps = int(dur/dt)

        Gposes = []
        Gweights = []

        Gpos = np.zeros(steps)
        Gweight = np.zeros(steps)

        t = np.arange(0, dur, dt)
        Gweight[int(np.round(0.0/dt)):int(np.round(30.0/dt))] = 1.0
        Gpos[int(np.round(0.0/dt)):int(np.round(1.0/dt))] = -180.
        Gpos[int(np.round(1.0/dt)):int(np.round(17.0/dt))] = -180+22.5*np.arange(0,16, dt)
        Gpos[int(np.round(17.0/dt)):int(np.round(30.0/dt))] = 180-22.5*np.arange(0,13, dt)
        # Gpos[int(np.round(30.0/dt)):int(np.round(31.0/dt))] = -180+22.5*2
        Gposes.append(Gpos)
        Gweights.append(Gweight)

        Gpos = np.zeros(steps)
        Gweight = np.zeros(steps)
        Gweight[int(np.round(0.0/dt)):int(np.round(33.0/dt))] = 0.4
        Gpos[int(np.round(0.0/dt)):int(np.round(33./dt))] = 60.

        Gposes.append(Gpos)
        Gweights.append(Gweight)

        screen = np.zeros((steps,360))
        x = np.arange(-180,180)
        for Gpos, Gweight in zip(Gposes, Gweights):
            for i in range(steps):
                pos = Gpos[i]
                screen[i, (np.arange(pos-10,pos+10, dtype = np.int32)+180)%360] += Gweight[i]
        screen = np.minimum(screen, 1)

        bilateral_PB_input_rate = {}

        for i, uid in enumerate(input_mapping):
            rr = np.zeros(steps)
            ranges = input_mapping[uid]
            for Gpos, Gweight in zip(Gposes, Gweights):
                rr += Gweight*sum([np.bitwise_and(Gpos > edge[0], Gpos <= edge[1]).astype(np.double) for edge in ranges])
            bilateral_PB_input_rate[uid] = np.minimum(rr, 1) * PB_bilateral_input_rate

        input_array = np.zeros((steps, len(new_uids)), np.bool)
        n = 0
        for uid in new_uids:
            input_array[:,n] = np.random.rand(steps) < bilateral_PB_input_rate[uid]*dt
            n += 1

        spike_time, spike_index = np.where(input_array)
        spike_time = spike_time * dt

        data = {'time': spike_time, 'index': spike_index}
        super(PB_Su_Bilateral_InputProcessor, self).__init__(
                    {'spike_state': {'uids': new_uids, 'data': data,
                                     'components': {'class': 'SynapseAMPA',
                                                    'gmax': 2.1e-3, 'st': 20.,
                                                    'reverse': 0.}
                                    },
                    },
                    input_file = None)#record_file,
                    # input_interval = record_interval)
                    #sensory_file = self.video.record_file,
                    #sensory_interval = self.video.record_interval,)
        self.to_file(record_file)

class EB_Su_rEPG_InputProcessor(PresynapticInputProcessor):
    def __init__(self, shape, dt, dur, name, video_config, rf_config, neurons,
                 scale = 1.0,
                 record_file = None, record_interval = 1):
        ring_input_rate = 200.
        steps = int(dur/dt)
        rEPG_input_rate = np.zeros(steps) + ring_input_rate

        input_array = np.zeros((steps, len(neurons)), np.bool)
        for i in range(len(neurons)):
            input_array[:,i] = np.random.rand(steps) < rEPG_input_rate*dt
        uids = list(neurons.keys())
        spike_time, spike_index = np.where(input_array)
        spike_time = spike_time * dt

        data = {'time': spike_time, 'index': spike_index}
        super(EB_Su_rEPG_InputProcessor, self).__init__(
                    {'spike_state': {'uids': uids, 'data': data,
                                     'components': {'class': 'SynapseAMPA',
                                                    'gmax': 0.5e-3, 'st': 20.,
                                                    'reverse': 0.}
                                    },
                    },
                    input_file = None)#record_file,
                    # input_interval = record_interval)
                    #sensory_file = self.video.record_file,
                    #sensory_interval = self.video.record_interval,)
        self.to_file(record_file)



class CX_Video(object):
    """
    Create a test video signal.
    """
    def __init__(self, shape, dt, dur, record_file = None, record_interval = 1):
        self.shape = shape
        self.dt = dt
        self.dur = dur
        self.N_t = int(self.dur/self.dt)
        self.frame = 0
        self.record_file = record_file
        self.record_interval = record_interval
        self.record_count = 0

    def run_step(self):
        frame = self.generate_frame()
        self.frame += 1
        if self.record:
            if self.record_count == 0:
                self.record_frame()
            self.record_count = (self.record_count+1)%self.record_interval
        return frame

    def pre_run(self):
        if self.record_file is not None:
            self.file = h5py.File(self.record_file, 'w')
            self.file.create_dataset('sensory',
                                     (0, self.shape[0], self.shape[1]),
                                     dtype = np.double,
                                     maxshape=(None, self.shape[0],
                                               self.shape[1]))
            self.record = True
        else:
            self.record = False
        self.data = np.empty(self.shape, np.double)

    def record_frame(self):
        self.file['sensory'].resize((self.file['sensory'].shape[0]+1,
                                         self.shape[0], self.shape[1]))
        self.file['sensory'][-1,:,:] = self.data

    def generate_frame(self):
        pass

    def close_file(self):
        if self.record:
            self.file.close()

    def __del__(self):
        try:
            self.close_file()
        except:
            pass

class moving_bar_l2r(CX_Video):
    def __init__(self, shape, dt, dur, bar_width, start = None, stop = None,
                 record_file = None, record_interval = 1):
        super(moving_bar_l2r, self).__init__(shape, dt, dur,
                                             record_file = record_file,
                                             record_interval = record_interval)
        self.bar_width = bar_width
        if start is None:
            self.start = 0.0
        else:
            self.start = start
        if stop is None:
            self.stop = dur
        else:
            self.stop = stop
        self.N_t = int((self.stop-self.start)/self.dt)

    def generate_frame(self):
        if self.start <= self.frame*self.dt < self.stop:
            start = int(np.ceil((self.frame-self.start/self.dt)*(self.shape[1]-self.bar_width)/float(self.N_t)))
            self.data.fill(0)
            self.data[:, start:start+self.bar_width] = 1.0
        else:
            self.data.fill(0)
        return self.data

class moving_bar_r2l(CX_Video):
    def __init__(self, shape, dt, dur, bar_width, start = None, stop = None,
                 record_file = None, record_interval = 1):
        super(moving_bar_r2l, self).__init__(shape, dt, dur,
                                             record_file = record_file,
                                             record_interval = record_interval)
        self.bar_width = bar_width
        if start is None:
            self.start = 0.0
        else:
            self.start = start
        if stop is None:
            self.stop = dur
        else:
            self.stop = stop
        self.N_t = int((self.stop-self.start)/self.dt)

    def generate_frame(self):
        if self.start <= self.frame*self.dt < self.stop:
            start = int(np.ceil((self.frame-self.start/self.dt)*(self.shape[1]-self.bar_width)/float(self.N_t)))
            self.data.fill(0)
            self.data[:, self.shape[1]-self.bar_width-start:-start] = 1.0
        else:
            self.data.fill(0)
        return self.data

class moving_bar_test(CX_Video):
    def __init__(self, shape, dt, dur, bar_width, start = None, stop = None,
                 record_file = None, record_interval = 1):

        super(moving_bar_test, self).__init__(shape, dt, dur,
                                             record_file = record_file,
                                             record_interval = record_interval)
        self.bar_width = bar_width
        if start is None:
            self.start = 0.0
        else:
            self.start = start
        if stop is None:
            self.stop = dur
        else:
            self.stop = stop
        self.N_t = int((self.stop-self.start)/self.dt)

        Gposes = []
        Gweights = []

        steps = int(35/dt)
        Gpos = np.zeros(steps)
        Gweight = np.zeros(steps)

        # t = np.arange(0, dur, dt)
        Gweight[int(np.round(0.0/dt)):int(np.round(30.0/dt))] = 1.0
        Gpos[int(np.round(0.0/dt)):int(np.round(1.0/dt))] = -180.
        Gpos[int(np.round(1.0/dt)):int(np.round(17.0/dt))] = -180+22.5*np.arange(0,16, dt)
        Gpos[int(np.round(17.0/dt)):int(np.round(30.0/dt))] = 180-22.5*np.arange(0,13, dt)
        # Gpos[int(np.round(30.0/dt)):int(np.round(31.0/dt))] = -180+22.5*2
        Gposes.append(Gpos)
        Gweights.append(Gweight)

        # Gpos = np.zeros(steps)
        # Gweight = np.zeros(steps)
        # Gweight[int(np.round(0.0/dt)):int(np.round(1.0/dt))] = 1.0
        # Gpos[int(np.round(0.0/dt)):int(np.round(1.0/dt))] = 40.
        # Gposes.append(Gpos)
        # Gweights.append(Gweight)

        Gpos = np.zeros(steps)
        Gweight = np.zeros(steps)
        Gweight[int(np.round(0.0/dt)):int(np.round(33.0/dt))] = 0.4
        Gpos[int(np.round(0.0/dt)):int(np.round(33.0/dt))] = 60.

        Gposes.append(Gpos)
        Gweights.append(Gweight)

        screen = np.zeros((int(35/self.dt),360))
        x = np.arange(-180,180)
        for Gpos, Gweight in zip(Gposes, Gweights):
            for i in range(steps):
                pos = Gpos[i]
                screen[i, (np.arange(pos-10,pos+10, dtype = np.int32)+180)%360] += Gweight[i]
        self.screen = np.minimum(screen, 1)[:,::-1]
        # with h5py.File('screen_1.h5', 'r') as f:
        #     # f.create_dataset('/data', data=self.screen)
        #     self.screen = f['data'][:][:,::-1]

    def generate_frame(self):
        self.data[:] = np.tile(self.screen[self.frame, :].reshape(1,-1), [self.shape[0],1])
        return self.data

def Video_factory(video_class_name):
    all_video_cls = CX_Video.__subclasses__()
    all_video_names = [cls.__name__ for cls in all_video_cls]
    try:
        video_cls = all_video_cls[all_video_names.index(video_class_name)]
    except ValueError:
        print('Invalid Video subclass name: {}'.format(video_class_name))
        print('Valid names: {}'.format(all_video_names))
        return None
    return video_cls

class CircularGaussianFilterBank(object):
    """
    Create a bank of circular 2D Gaussian filters.

    Parameters
    ----------
    shape : tuple
        Image dimensions.
    sigma : float
        Parameter of Gaussian.
    n : int
        How many blocks should occupy the x-axis.
    """

    def __init__(self, shape, sigma, n, hemisphere = 'L'):
        self.shape = shape
        self.sigma = sigma
        self.n = n
        self.hemisphere = hemisphere

        # Compute maximal and minimal response of a centered filter to use for
        # normalization:
        self.norm_min = np.inner(np.zeros(np.prod(shape)),
                                 self.gaussian_mat(shape, sigma, 0, 0, n).reshape(-1))
        self.norm_max = np.inner(np.ones(np.prod(shape)),
                                 self.gaussian_mat(shape, sigma, 0, 0, n).reshape(-1))

    def normalize_output(self, output):
        """
        Normalize filter output against range of responses to a centered RF.
        """

        return output/(self.norm_max-self.norm_min)

    @classmethod
    def func_gaussian(cls, x, y, sigma):
        """
        2D Gaussian function.
        """

        return (1.0/(1*np.pi*(sigma**2)))*np.exp(-(1.0/(2*(sigma**2)))*(x**2+y**2))

    @classmethod
    def gaussian_mat(cls, shape, sigma, n_x_offset, n_y_offset, n):
        """
        Compute offset circular 2D Gaussian.
        """

        # Image dimensions in pixels:
        N_y, N_x = shape

        # Normalized image width and height:
        x_max = 1.0
        y_max = N_y/float(N_x)

        X, Y = np.meshgrid(np.linspace(-x_max/2, x_max/2, N_x)-(n_x_offset/float(n)),
                           np.linspace(-y_max/2, y_max/2, N_y)-(n_y_offset/float(n)/2))
        return cls.func_gaussian(X, Y, sigma)

    def create_filters(self, filename = None):
        """
        Create filter bank as order-4 tensor.
        """

        N_y, N_x = self.shape

        # Normalized image width and height:
        x_max = 1.0
        y_max = N_y/float(N_x)

        # Compute how many blocks to use along the y-axis:
        n = self.n
        m = n*2*N_y//N_x

        # Construct filters offset by the blocks:
        # n_x_offsets = np.linspace(np.ceil(-n/2.0), np.floor(n/2.0), n)
        if self.hemisphere == 'L':
            n_x_offsets = np.linspace((-n+0.5)/2, -0.5/2, n)
        else:
            n_x_offsets = np.linspace(0.5/2, (n-0.5)/2, n)[::-1].copy()
        n_y_offsets = np.linspace(np.ceil(-m/2.0), np.floor(m/2.0), m)
        filters = np.empty((m, n, N_y, N_x), np.float64)
        for j, n_x_offset in enumerate(n_x_offsets):
            for i, n_y_offset in enumerate(n_y_offsets):
                filters[i, j] = self.gaussian_mat(self.shape, self.sigma,
                                                 n_x_offset, n_y_offset, n)
        self.filters = filters
        if filename is not None:
            file = h5py.File(filename,'w')
            file.create_dataset('filter', filters.shape, filters.dtype, data=filters)
            file.close()

    def apply_filters(self, frame, normalize = True, scale = 1.0):
        """
        Compute inner products of computed filters and a video frame.
        """

        result = np.tensordot(self.filters, frame)
        if normalize:
            return self.normalize_output(result)*scale
        else:
            return result*scale

class RectangularFilterBank(object):
    """
    Create a bank of 2D rectangular filters that tile the x-axis.
    """

    def __init__(self, shape, n, scale = 1.0):
        self.shape = shape
        self.n = n

        # Compute maximal and minimal response of a centered filter to use for
        # normalization:
        self.norm_min = np.inner(np.zeros(np.prod(shape)),
                                 self.rect_mat(shape, 0, n).reshape(-1))
        self.norm_max = np.inner(np.ones(np.prod(shape)),
                                 self.rect_mat(shape, 0, n).reshape(-1))

    def normalize_output(self, output):
        """
        Normalize filter output against range of responses to a centered RF.
        """

        return output/(self.norm_max-self.norm_min)

    @classmethod
    def func_rect(cls, x, y, width):
        xx = x - (x>0.5) + (x<=-0.5)
        return np.logical_and(xx > -width/2.0, xx <= width/2.0).astype(np.float64)

    @classmethod
    def rect_mat(cls, shape, n_x_offset, n):
        N_y, N_x = shape

        x_max = 1.0
        y_max = N_y/float(N_x)

        X, Y = np.meshgrid(np.linspace(-x_max/2, x_max/2, N_x)-(n_x_offset/float(n)),
                           np.linspace(-y_max/2, y_max/2, N_y))
        return cls.func_rect(X, Y, 1.0/n)

    def create_filters(self):
        N_y, N_x = self.shape

        # Normalized image width and height:
        x_max = 1.0
        y_max = N_y/float(N_x)

        # Construct filters offset by the blocks:
        n_x_offsets = np.linspace(np.ceil(-self.n/2.0), np.floor(self.n/2.0), self.n+1)[:-1]
        filters = np.empty((self.n, N_y, N_x), np.float64)

        for j, n_x_offset in enumerate(n_x_offsets):
            filters[j] = self.rect_mat(self.shape, n_x_offset, self.n)
        self.filters = filters
        # file = h5py.File('PB_filters.h5','w')
        # file.create_dataset('filter', filters.shape, filters.dtype, data=filters)
        # file.close()

    def apply_filters(self, frame, normalize = True, scale = 1.0):
        """
        Compute inner products of computed filters and a video frame.
        """

        result = np.tensordot(self.filters, frame)
        if normalize:
            return self.normalize_output(result)*scale
        else:
            return result*scale
