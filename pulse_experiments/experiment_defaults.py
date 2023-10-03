"""Common code for all experiments.

Note `https://docs.python.org/3/tutorial/modules.html` specifies how to properly do
sibling imports, and is prefered to using `sys.path.append`.  Unfortunately, this
requires running the interpreter from a parent directory containing this repo. This was
not ideal for my workflow - I would rather run my experiments from the directory
containing the script. Users of this package will not have this issue, and avoid this.
"""

import sys
import os.path

dir_path = os.path.dirname(__file__)
for module_path in ['neural_field_synaptic_depression', 'plotting_helpers']:
    sys.path.append(os.path.join(dir_path, '..', module_path))

media_path = os.path.join(dir_path, 'media')
data_path = os.path.join(dir_path, 'data')
