# CXcircuits
FlyBrainLab Circuit Library for the Central Complex

### Installation

Note: the package must be installed both in the environment hosting the FlyBrainLab user-side components and in the environment hosting the backend servers.
For example, if you are running FlyBrainLab using the [full installation](https://github.com/FlyBrainLab/FlyBrainLab#12-full-installation) or
[Docker image](https://github.com/FlyBrainLab/FlyBrainLab#13-docker-image), you only need to install it in the `ffbo` environment since both
FlyBrainlab user-side components and backend servers are installed within the same environment. However, if you are running
the FlyBrainLab [user-side components](https://github.com/FlyBrainLab/FlyBrainLab#11-installing-only-user-side-components)
on a separate machine, you must install this package in both places.


```bash
git clone https://github.com/FlyBrainLab/CXcircuits.git
cd CXcircuits
python setup.py develop
```

### Loading Models to NeuroArch Database

Models are store in each of the subdirectories in the folder `models`. Go into each subdirectories and run 
```bash
python load_database.py
```

Each model is accompanied by a notebook in the `notebooks/elife20` directory. After loading the database,
open the corresponding notebook in FlyBrainLab and run through the notebook.



