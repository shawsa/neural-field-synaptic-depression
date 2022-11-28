
import sys
import os.path

dir_path = os.path.dirname(__file__)
for module_path in ['neural_field_synaptic_depression', 'plotting_helpers']:
    sys.path.append(os.path.join(dir_path, '..', module_path))

media_path = os.path.join(dir_path, 'media')
data_path = os.path.join(dir_path, 'data')
