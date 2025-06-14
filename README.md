# ED(Education Distillation)
This repo covers the implementation of the following KSEM 2025 paper:
Education Distillation: Let the Model Learn in the School

## Installation
This repo was tested with Python3.8, CUDA11.5, Pytorch1.10.1

## Running
1. Train teacher models
``` shell
python train_teacher.py 
```

2. Distill student model
``` shell
python train_student.py
```
where the flags are explained as:
* `--distill`: specify the distillation method, e.g. `kd`, `hint`
* `--model_s`: specify the student model, see 'models/__init__.py' to check the available model types.
* `--teacher_num`: specify the ensemble size (number of teacher models)

## Citation
If you find this repository useful, please consider citing the following paper:
```
@article{Feng2025,
  title={Education Distillation: Let the Model Learn in the School},
  author={Ling Feng, Tianhao Wu, Xiangrong Ren, Zhi Jing, and Xuliang Duan},
  journal={arXiv preprint arXiv:2311.13811​​},
  year={2025}
}

```
