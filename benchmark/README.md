# IXPUG_DNN_benchmark
Scripts for benchmarking dnn frameworks for IXPUG report

Clone this repository to destination machine.

## Get networks

```bash
 <openvino_dir>/bin/setupvars.sh
 python3 <openvino_dir>/deployment_tools/tools/model_downloader/downloader.py --output <output_folder> -- name resnet-50 
 python3 <openvino_dir>/deployment_tools/tools/model_downloader/downloader.py --output <output_folder> -- name ssd300
 
 python3 <openvino_dir>/deployment_tools/model_optimizer/mo.py --input_model <resnet50_folder>/resnet-50.caffemodel --input_proto <resnet50_folder>/resnet-50.prototxt 
 
 python3 <openvino_dir>/deployment_tools/model_optimizer/mo.py --input_model <ssd300_folder>/ssd300.caffemodel --input_proto <ssd300_folder>/ssd300.prototxt  --mean_values [104.0,117.0,123.0]
 
```

## Prepare software


The easiest way to avoid dependency hell is using seperate environments created by Conda. 

Download Miniconda here [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) and install.

Create new environment for every framework, for example
```bash
 conda create -n caffe [python=3.6]
```

Next, activate environment

```bash
 conda activate caffe
```

Next, install framework and run benchmark.


## Caffe

Variant 1. Install caffe from conda

```bash
 conda create -n caffe
 conda activate caffe
 conda install -c intel caffe
```

### Start benchmark classifiction

```bash
 conda activate caffe
 cd <IXPUG_DNN_benchmark>/caffe_benchmark
 mkdir results_classification
 python3 caffe_benchmark.py -t classification -i ../datasets/imagenet/ -p ../models/resnet-50.prototxt -m ../models/resnet-50.caffemodel -ni 1000 -o False -of ./results_classification/ -r ./results_classification/result.csv
```

### Start benchmark detection
```bash
 conda activate caffe
 cd <IXPUG_DNN_benchmark>/caffe_benchmark
 mkdir results_detection
 python3 caffe_benchmark.py -t detection -i ../datasets/pascal_voc/ -p ../models/ssd300.prototxt -m ../models/ssd300.caffemodel -ni 1000 -o False -of ./results_detection/ -r ./results_detection/result.csv -me [104,117,123]
 
```

## OpenCV

Variant 1. Install opencv from conda

```bash
 conda create -n opencv
 conda activate opencv
 conda install -c conda-forge opencv
```

### Start benchmark classifiction

```bash
 conda activate opencv
 cd <IXPUG_DNN_benchmark>/opencv_benchmark
 mkdir results_classification
 python3 opencv_benchmark.py -i ../datasets/imagenet/ -p ../models/resnet-50.prototxt -m ../models/resnet-50.caffemodel -ni 1000 -of ./results_classification/ -r ./results_classification/result.csv -w 224 -he 224 -s 1.0
```

if you want to save images output add argument `-o True`.

### Start benchmark detection

```bash
 conda activate opencv
 cd <IXPUG_DNN_benchmark>/opencv_benchmark
 mkdir results_detection
 python3 opencv_benchmark.py -t detection -i ../datasets/pascal_voc/ -p ../models/ssd300.prototxt -m ../models/ssd300.caffemodel -ni 1000 -of ./results_detection/ -r ./results_detection/result.csv -w 300 -he 300 -s 1.0
```

## OpenVINO

Variant 1. Install from off. site


### Optimization 
To get better performance, try different `-tn`, `-sn`, `-rn` and `-b` parameters, and set `-o` parameter (output) to `False`.

### Start benchmark classifiction

```bash
 conda activate openvino
 cd <IXPUG_DNN_benchmark>/openvino_benchmark
 mkdir result_sync_classification
 mkdir result_async_classification
 
 
 python3 openvino_benchmark_sync.py -i ../datasets/imagenet/ -c ../models/resnet-50.xml -m ../models/resnet-50.bin -ni 1000 -o False -of ./result_sync/ -r result_sync.csv -s 1.0 -w 224 -he 224 -b 1

 python3 openvino_benchmark_async.py -i ../datasets/imagenet/ -c ../models/resnet-50.xml -m ../models/resnet-50.bin -ni 1000 -o False -of ./result_async/ -r result_async.csv -s 1.0 -w 224 -he 224 -b 1
```

### Start benchmark detection

```bash
 conda activate openvino
 cd <IXPUG_DNN_benchmark>/openvino_benchmark
 mkdir result_sync_detection
 mkdir result_async_detection
 
python3 openvino_benchmark_sync.py -t detection -i ../datasets/pascal_voc/ -c ../models/ssd300.xml -m ../models/ssd300.bin -ni 1000 -of ./result_sync_detection/ -r ./result_sync_detection/result_sync.csv -s 1.0 -w 300 -he 300 -b 1 -e ~/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so

python3 openvino_benchmark_async.py -t detection -i ../datasets/pascal_voc/ -c ../models/ssd300.xml -m ../models/ssd300.bin -ni 1000 -of ./result_async_detection/ -r ./result_async_detection/result_async.csv -s 1.0 -w 300 -he 300 -b 1 -e ~/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so
```
