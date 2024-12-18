# Longitudinal infant functional connectivity prediction via conditional intensive triplet network

### Framework:
![framework](main.png)

### Papers:
This repository provides a PyTorch implementation of the models adopted in the following papers:

- Yu, X., et al. "Longitudinal infant functional connectivity prediction via conditional intensive triplet network." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 20202.


### Code:
#### main_pcc_mae.py
You need to run this file to start. The hyper-parameters of loss weight can be defined in this file.

Run `python ModelCode\main_pcc_mae.py`. 

#### NNFunctions.py
We created a triple construction method in this file.



Tested with:
- PyTorch 1.9.0
- Python 3.7.0

### Data:
We used 464 subjects from BCP dataset in our research. For each subject, we generated the funcational connectivity (FC) matrices for infants.

### Citation:
If you used the code or data of this project,  please cite:
    
    @inproceedings{yu2022longitudinal,
    title={Longitudinal infant functional connectivity prediction via conditional intensive triplet network},
    author={Yu, Xiaowei and Hu, Dan and Zhang, Lu and Huang, Ying and Wu, Zhengwang and Liu, Tianming and Wang, Li and Lin, Weili and Zhu, Dajiang and Li, Gang},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    pages={255--264},
    year={2022},
    organization={Springer}
    }

