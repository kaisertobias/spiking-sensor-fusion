# spiking-sensor-fusion
This directory contains algorithms for heterogeneous sensor data fusion, which were implemented in the Nengo framework. It is a fork of https://github.com/th-nuernberg/spicenet.

## KalmanFilter
In this subdirectory a linear Kalman filter, an Extended Kalman filter and an Unscented Kalman filter are implemented.

## SPICEnet
In this subdirectory a processing paradigm for extracting correlations from sensorimotor streams, called SPICEnet, is implemented. Corresponding publications:
1. Axenie, Cristian, and Jörg Conradt. "Learning sensory correlations for 3D egomotion estimation." Biomimetic and Biohybrid Systems: 4th International Conference, Living Machines 2015, Barcelona, Spain, July 28-31, 2015, Proceedings 4. Springer International Publishing, 2015. https://link.springer.com/chapter/10.1007/978-3-319-22979-9_32
2. Axenie, Cristian, Christoph Richter, and Jörg Conradt. "A self-synthesis approach to perceptual learning for multisensory fusion in robotics." Sensors 16.10 (2016): 1751. https://www.mdpi.com/1424-8220/16/10/1751