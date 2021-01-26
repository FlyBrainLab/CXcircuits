import re
import os
from .parse_arborization import NeuronArborizationParser

folder = os.path.dirname(os.path.abspath(__file__))

def get_subregion_loc(svg = "{}/../img/cx.svg".format(folder)):
	subregion_to_location = {}
	subregion_translation = {'PB':(698.03863,215.29411),
							'EB':(390.00293,232.57211),
							'no':(450.73004,343.10509),
							'lal':(434.70346,343.10509),
							'NO':(434.70346,343.10509),
							'FB':(262.362,-40.400391),
							'LAL':(434.70346,343.10509),
							'BU':(0,0),
							'bu':(0,0)}

	with open(svg,'r') as file:
		record_switch = False
		subregion_switch = False
		# subregion_width = ''
		# subregion_height = ''
		# subregion_x = ''
		# subregion_y = ''

		for line in file:
			if '<path' in line or '<rect' in line:
				record_switch = True
			if record_switch==True:
				if 'class="region"' in line:
					subregion_switch = True
				if 'width=' in line:
					subregion_width = re.findall('width="(.*?)"',line)[0]
				if 'height=' in line:
					subregion_height = re.findall('height="(.*?)"',line)[0]
				if 'x=' in line:
					subregion_x = re.findall('x="(.*?)"',line)[0]
				if 'y=' in line:
					subregion_y = re.findall('y="(.*?)"',line)[0]
				if 'label=' in line:
					subregion_label = re.findall('label="(.*?)"',line)[0]
					region_label = subregion_label.split('/')[0]
			if '/>' in line:
				record_switch = False
				if subregion_switch == True:
					x_adjust = 0
					y_adjust = 0
					if float(subregion_x) < 0:
						x_adjust = float(subregion_width)
						# print subregion_label
					if float(subregion_y) < 0:
						y_adjust = float(subregion_height)
						# print subregion_label
					subregion_to_location[subregion_label] = [subregion_width, subregion_height, str(abs(float(subregion_x))+subregion_translation[region_label][0] - x_adjust), str(abs(float(subregion_y))+subregion_translation[region_label][1] - y_adjust)]
					subregion_switch = False
	return subregion_to_location

subregion_to_location = get_subregion_loc()
# parser = NeuronArborizationParser()
# parsed = parser.parse(new_neuron)

def convert_into_list(parsed):
    result = []
    presynaptic = []
    postsynaptic = []

    for d in parsed:
        neuropil = d['neuropil']
        neurite = list(d['neurite'])[0]
        for region in list(d['regions']):
            if isinstance(region, str):
                region_str = neuropil+'/'+str(region)
            elif isinstance(region, tuple):
                region_str = neuropil+'/('
                for item in region:
                    region_str = region_str+item+','
                region_str = region_str[:-1] + ')'
            result.append(region_str)
            if neurite=='b':
                presynaptic.append(region_str)
            elif neurite=='s':
                postsynaptic.append(region_str)
    return result, presynaptic, postsynaptic

def tuple_to_string(region):
	region_str = '('
	for item in region:
		region_str = region_str + item + ','
	region_str = region_str[:-1] + ')'
	return region_str

def find_center(loc):
	center = (float(loc[2]) + float(loc[0])/2, float(loc[3]) + float(loc[1])/2)
	return center

def find_loc_to_connect(parsed):
	loc_to_connect = []
	post_loc = []
	pre_loc = []
	for region in parsed:
		region_label = region['neuropil']
		region_neurite = list(region['neurite'])[0]
		for subregion in region['regions']:
			if isinstance(subregion,tuple):
				subregion_label = region_label + '/' + tuple_to_string(subregion)
			else:
				subregion_label = region_label + '/' + subregion
			subregion_loc = find_center(subregion_to_location[subregion_label])
			if region_neurite == 's':
				post_loc.append(subregion_loc)
			elif region_neurite == 'b':
				pre_loc.append(subregion_loc)
	for post_loc_item in post_loc:
		for pre_loc_item in pre_loc:
			loc_to_connect.append((post_loc_item,pre_loc_item))
	return loc_to_connect

def tmp_neuron(loc_to_connect, parsed_list, presynaptic, postsynaptic):
    tmp_str = '<g\n'  + 'id="tmp_neuron"\n'
    # add pattern attribution
    tmp_str = tmp_str + "pattern=\""
    for pattern_single in parsed_list:
        tmp_str = tmp_str + pattern_single+' '
    tmp_str = tmp_str[:-1] + "\"\n"
    # add presynaptic attribution
    tmp_str = tmp_str + "presynaptic=\""
    for pre_single in presynaptic:
        tmp_str = tmp_str + pre_single+' '
    tmp_str = tmp_str[:-1] + "\"\n"
    # add postsynaptic attribution
    tmp_str = tmp_str + "postsynaptic=\""
    for post_single in postsynaptic:
        tmp_str = tmp_str + post_single+' '
    tmp_str = tmp_str[:-1] + "\">\n"

    tmp_index = 0
    for loc_pair in loc_to_connect:
        path = '<path\nid="tmp{}"\nd="M {},{} L {},{}"\nstyle="opacity:1;vector-effect:none;fill:none;fill-opacity:1;stroke:#ffffff;stroke-width:2px;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />'.format(tmp_index, loc_pair[0][0], loc_pair[0][1], loc_pair[1][0], loc_pair[1][1])
        tmp_index = tmp_index + 1
        # print path
        tmp_str = tmp_str + path
    tmp_str = tmp_str + '</g>'
    return tmp_str

def generate_svg(neurons, fin='{}/../img/cx.svg'.format(folder),
                 fout='{}/../img/cx_tmp.svg'.format(folder)):
    parser = NeuronArborizationParser()
    tmp_strs = []
    for neuron in neurons:
        parsed = parser.parse(neuron)
        parsed_list, presynaptic, postsynaptic = convert_into_list(parsed)
        loc_to_connect = find_loc_to_connect(parsed)
        tmp_str = tmp_neuron(loc_to_connect, parsed_list, presynaptic, postsynaptic)
        tmp_strs.append(tmp_str)

    with open(fin,'r') as f_in:
        with open(fout,'w') as f_out:
            for line in f_in:
                if '</svg>' in line:
                    break
                f_out.write(line)
            for tmp_str in tmp_strs:
                f_out.write(tmp_str)
            f_out.write('\n</svg>')


# with open("./img/cx_final_tmp.svg",'wb') as file:
if __name__ == "__main__":
    new_neuron = ["PB/L9/s-LAL/RGT/b-FB/(1,L1)/b"]
    generate_svg(new_neuron)
