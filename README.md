Mixed-precision QNN co-design tool for MCU
===================================
This is a HW-SW co-design tool build on top of DORY project.
The tool guides the user from the design of the quantized neural networks up to the deployment
on the target device. 

*NOTE*: The code is currently anonymized because the related paper is under review. Part of the code refers to the SoA Dory project, which we are not part of, and hence it could not be anonymized.

Installation
------------

1. Clone the repository and the required submodules.
2. Build the Dockerfile with the required SDKs and Python env
```
cd ALADIN
docker buildx build -t dory-docker:3.9 ./.devcontainer/ 
```

3. Once connected to the terminal of the container run the following commands:
```
source /dory_env/bin/activate
source docker_util/docker_pulp_sdk.sh
```
*NOTE: these packages cannot be installed from the Dockerfile*

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
This command will generate a JSON file where the user can specify the information for the quantization process.

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
python impl_design.py --config_files <list of config files> 
cd ..
```
This will generate the CSV and plots realtively to the implementation information

4. Run implementation design search:
```
source docker_util/docker_pulp_sdk.sh
python platform_design/design_search.py --config_files <list of config files> 
```
This will generate the CSV and plots realtively to the platform performances
 
*NOTE: you can find config files already available in the repo*


### Reference
*We are NOT the developers of DORY project*, however if you are interested please consider to cite also their paper: https://ieeexplore.ieee.org/document/9381618 (preprint available also at https://arxiv.org/abs/2008.07127)
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
This project and DORY are released under Apache 2.0, see the LICENSE file in the root of this repository for details.
