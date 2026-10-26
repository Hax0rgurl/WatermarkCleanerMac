ProPainter by Shangchen Zhou, Chongyi Li, Kelvin C.K. Chan, Chen Change Loy, S-Lab, Nanyang Technological University.
Source: https://github.com/sczhou/ProPainter
Pinned revision: e870e79321c31b733e2031af5aa2fb1fe3ac7eec
License: S-Lab License 1.0, full text in LICENSE. Non-commercial redistribution and use; contact upstream for commercial permission.
Changes: package-relative imports; unused training and optional CUDA correlation paths removed; verified local checkpoint loading only. Original pretrained inference architecture retained. Meshgrid indexing and disabled mixed precision use explicit current PyTorch APIs with unchanged numerical behavior; the base deformable layer has a working Torchvision forward implementation.

The RAFT optical-flow modules are from Princeton Vision & Learning Lab, copyright 2020 princeton-vl, BSD 3-Clause. Full upstream notice is retained in raft/LICENSE; source https://github.com/princeton-vl/RAFT.
