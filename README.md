SPOT-RNA-1D: *RNA backbone torsion and pseudotorsion angle prediction using dilated convolutional neural networks.*
====

System Requirments
----

**Hardware Requirments:**
SPOT-RNA-1D predictor requires only a standard computer with around 32 GB RAM to support the in-memory operations for RNAs sequence length less than 2000.

**Software Requirments:**
* [Python3](https://docs.python-guide.org/starting/install3/linux/)
* [virtualenv](https://virtualenv.pypa.io/en/latest/installation/) or [Anaconda](https://anaconda.org/anaconda/virtualenv)
* [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) (Optional If using GPU)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn) (Optional If using GPU)

SPOT-RNA-1D has been tested on Ubuntu 14.04, 16.04, and 18.04 operating systems.


Installation
----

To install SPOT-RNA-1D and it's dependencies following commands can be used in terminal:

1. `git clone https://github.com/jaswindersingh2/SPOT-RNA-1D.git && cd SPOT-RNA-1D`
2. `wget -O checkpoints.tar.xz 'https://www.dropbox.com/s/r9hp20gk30unptf/checkpoints.tar.xz' || wget -O checkpoints.tar.xz 'https://app.nihaocloud.com/f/4d2385c633554ccaa85c/?dl=1'`
3. `tar -xvf checkpoints.tar.xz && rm checkpoints.tar.xz`

Either follow **virtualenv** column steps or **conda** column steps to create virtual environment and to install SPOT-RNA-1D dependencies given in table below:<br />

|  | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; virtualenv | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; conda |
| :- | :-------- | :--- |
| 4. | `virtualenv -p python3.6 venv` | `conda create -n venv python=3.6` |
| 5. | `source ./venv/bin/activate` | `conda activate venv` | 
| 6. | *To run SPOT-RNA-1D on CPU:*<br /> <br /> `pip install tensorflow==1.15.0` <br /> <br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *or* <br /> <br />*To run SPOT-RNA-1D on GPU:*<br /> <br /> `pip install tensorflow-gpu==1.15.0` | *To run SPOT-RNA-1D on CPU:*<br /> <br /> `conda install tensorflow==1.15.0` <br /> <br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *or* <br /> <br />*To run SPOT-RNA-1D on GPU:*<br /> <br /> `conda install tensorflow-gpu==1.15.0` |
| 7. | `pip install -r requirements.txt` | `while read p; do conda install --yes $p; done < requirements.txt` | 

Usage
----

**To run the SPOT-RNA-1D**

```
./run.py --seq_file inputs/TS1_seqs.fasta --save_outputs outputs/
```

Datasets
----

The following dataset was used for Training, Validation, and Testing of SPOT-RNA-1D:

[Dropbox](https://www.dropbox.com/s/fl1upqsvd7rpyrl/RNAsnap2_data.zip) or [Nihao Cloud](https://app.nihaocloud.com/f/afea8e005a964bf8bb0f/)

Citation guide
----

**If you use SPOT-RNA-1D for your research please cite the following paper:**

Singh, J., Paliwal, K., Singh, J., Zhou, Y., 2021. RNA backbone torsion and pseudotorsion angle prediction using dilated convolutional neural networks.

Licence
----
Mozilla Public License 2.0


Contact
----
jaswinder.singh3@griffithuni.edu.au, yaoqi.zhou@griffith.edu.au

