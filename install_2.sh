# in Makefile.config
'''
OPENCV_VERSION := 3

...

#INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
#LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/
'''

make -j4 all
make -j4 test
make -j4 runtest

# in line 253 of caffe/python/caffe/io.py
'''
if ms != self.inputs[in_][1:]:
    print(self.inputs[in_])
    in_shape = self.inputs[in_][1:]
    m_min, m_max = mean.min(), mean.max()
    normal_mean = (mean - m_min) / (m_max - m_min)
    mean = resize_image(normal_mean.transpose((1,2,0)),in_shape[1:]).transpose((2,0,1)) * (m_max - m_min) + m_min
    #raise ValueError('Mean shape incompatible with input shape.')
'''

make pycaffe
make distribute
# in ~/.bashrc: export PYTHONPATH=<caffe-home>/python:$PYTHONPATH
