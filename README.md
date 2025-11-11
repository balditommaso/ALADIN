Mixed-precision QNN co-design tool for MCU
===================================

DORY is an automatic tool to deploy DNNs on low-cost MCUs with typically less than 1MB of on-chip SRAM memory. 


Installation
------------

1. Clone the repository and the required submodules.
2. Build the Dockerfile with the required SDKs and Python env
```
cd mixed-precision-dory
docker build -t dory-docker:3.9 ./.devcontainer/ 
```

Experimets
---------
1. Train the network:
```
cd impl_design
python training.py --save_dir ./checkpoint \
  --dataset cifar-10 \
  --model mobilenet_v1 \
  --file_name MV1 \
  --lr 0.05 \
  --batch_size 128 \
  --num_workers 4 \
  --seed 42 \
  --epochs 300 \
  --scheduler cosine \
  --save_json
cd ..
```
This command will generate a JSON file where the user can specify the inforamtion for the quantization process.

2. Apply QAT to a network:
```
cd impl_design
python qat.py --save_dir ./checkpoint \
  --model_path ./checkpoint/MV1.ckpt \
  --dataset cifar-10 \
  --model mobilenet_v1 \
  --config_path ./checkpoint/MV1_mix_4_8bit.json \
  --file_name quant_MV1_4bit \
  --lr 0.001 \
  --batch_size 128 \
  --num_workers 4 \
  --seed 42 \
  --epochs 50 \
  --scheduler cosine \
  --save_onnx
cd ..
```
This command will generate the QONNX file that will be used from Dory for the parsing and it generate a JSON file that can be used by to specify the *implementation* choice of each operation.

3. Run implementation aware analysis:
```
cd impl_design
python impl_design.py ./checkpoint/<implementation info>.json \
  profile.csv
cd ..
```
This will generate the CSV and plots realtively to the implementation information

4. Run implementation design search:
```
source docker_util/docker_pulp_sdk.sh
python platform_design/design_search.py ./checkpoint/<implementation info>.json \
  profile.csv
```
This will generate the CSV and plots realtively to the platform performances
 


### Reference
Project build on top of DORY tool, please make sure to cite also their paper: https://ieeexplore.ieee.org/document/9381618 (preprint available also at https://arxiv.org/abs/2008.07127)
```
@article{burrello2020dory,
  author={A. {Burrello} and A. {Garofalo} and N. {Bruschi} and G. {Tagliavini} and D. {Rossi} and F. {Conti}},
  journal={IEEE Transactions on Computers}, 
  title={DORY: Automatic End-to-End Deployment of Real-World DNNs on Low-Cost IoT MCUs}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TC.2021.3066883}
}
```

### Contributors
+ **Tommaso Baldi**, *SSSA*, [email](mailto:tommaso.baldi@santannapisa.it)


### License
Project and DORY are released under Apache 2.0, see the LICENSE file in the root of this repository for details.
