# -*- coding: utf-8 -*-
"""
@author: Valentina Kustikova

Inference performance analysis for MXNet

Sample string to run benchmark: 

    python mxnet_inference.py -t classification -g resnet-50-symbol.json \
        -m resnet-50-0000.params -d "3 224 224" -i images_classification \
        -ni 16 -o true -l synset.txt -b 4
        
    python mxnet_inference.py -t detection -g ssd_300-symbol.json \
        -m ssd_300-0000.params -d "3 300 300" -i images_detection \
        -ni 16 -o true -l pascal_voc_names.txt -b 4

Last modified 24.10.2019
"""

import argparse
import os
import re
import sys
import inspect
import numpy as np
import mxnet as mx
import logging as log
import datetime
from time import time


def log_debug_message(message):
    ts = time()
    ts_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    func = inspect.currentframe().f_back.f_code
    log.debug("[%s] %s: %s(..) in %s:%i" % (ts_str, message, func.co_name,
        func.co_filename, func.co_firstlineno))
    
def log_error_message(message):
    ts = time()
    ts_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    func = inspect.currentframe().f_back.f_code
    log.error("[%s] %s: %s(..) in %s:%i" % (ts_str, message, func.co_name,
        func.co_filename, func.co_firstlineno))


def log_info_message(message):
    ts = time()
    ts_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    func = inspect.currentframe().f_back.f_code
    log.info("[%s] %s: %s(..) in %s:%i" % (ts_str, message, func.co_name,
        func.co_filename, func.co_firstlineno))


def build_argparser():
    log_debug_message('START')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', help = 'Path to the .json file \
        containing serialized computational graph', required = True,
        type = str)
    parser.add_argument('-m', '--model', help = 'Path to the .param file \
        with a trained weights', required = True, type = str)
    parser.add_argument('-d', '--data_shape', help = 'Input data shape \
        (for ResNet-50 it is \'3 224 224\')', required = True, type = str)
    parser.add_argument('-i', '--input_dir', help = 'Name of the directory \
        containing input images', required = True, type = str)
    parser.add_argument('-ni', '--number_iter', help = 'Number of inference \
        iterations', required = True, type = int)
    parser.add_argument('-o', '--need_output', help = 'Get output',
        required = True, type = lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-of', '--output_file', help = 'Name of the output file',
        default = 'res.txt', type = str)
    parser.add_argument('-rf', '--perf_file', help = 'Name of the file \
        containing performance results', default = 'result.csv', type = str)
    parser.add_argument('-t', '--task_type', help = 'Task type: \
        classification / detection', default = 'classification', type = str)
    parser.add_argument('-me', '--mean', help = 'Input mean values', 
        default = '[0 0 0]', type = str)
    parser.add_argument('-b', '--batch_size', help = 'batch size', 
        default = 1, required = True, type = int)
    parser.add_argument('-l', '--label_file', help = 'List of labels (text file)',
        default = None, type = str)
    
    log_debug_message('FINISH')
    return parser


def load_network(graph, model):
    log_debug_message('START')
    
    re_json = r'([\w\d\-]+)\-symbol.json'
    re_params = r'([\w\d\-]+)\-([\d]+).params'

    cre_json = re.compile(re_json)
    cre_params = re.compile(re_params)

    m_json = cre_json.match(graph)
    m_params = cre_params.match(model)

    json_model_name = m_json.group(1)
    params_model_name = m_params.group(1)

    epoch_num = int(m_params.group(2))
    if json_model_name != params_model_name:
        error_message = 'Incorrect .json and .params file names. \
            The format of file name is as follows: prefix-symbol.json and \
            prefix-epoch.params'
        log_error_message(error_message)
        raise Exception(error_message)
    log_debug_message('START: mx.model.load_checkpoint(..)')
    sym, args, aux = mx.model.load_checkpoint(json_model_name, epoch_num)
    mx_module = mx.mod.Module(symbol = sym, data_names = ['data'],
        context = mx.cpu())
    log_debug_message('FINISH: mx.model.load_checkpoint(..)')

    log_debug_message('FINISH')
    return mx_module, sym, args, aux


def load_images(input_dir, batch_size, data_name, data_shape):
    log_debug_message('START')
    
    log_debug_message('START: load images')
    image_files = [ os.path.join(input_dir, f) for f in os.listdir(input_dir) \
        if os.path.isfile(os.path.join(input_dir, f))]
    images = np.empty((len(image_files), data_shape['nchannels'],
        data_shape['width'], data_shape['height']), dtype = 'float32')
    for idx in range(len(image_files)):
        img = mx.image.imread(image_files[idx])
        img = mx.image.imresize(img, data_shape['width'], data_shape['height'])
        img = img.transpose((2, 0, 1)) # channels first
        img = img.astype(dtype = 'float32')
        images[idx] = img.asnumpy()
    log_debug_message('FINISH: load images')

    log_debug_message('START: сreate iterator')
    eval_data = mx.io.NDArrayIter({ data_name: images },
        label = None, batch_size = batch_size, shuffle = False,
        last_batch_handle = 'discard')
    log_debug_message('FINISH: сreate iterator')

    log_debug_message('FINISH')
    return image_files, eval_data

    
def mxnet_benchmark(task_type, graph, model, input_dir, number_iter,
        data_shape, batch_size = 1, need_output = False, label_file = '',
        output_file = 'res.txt'):
    log_debug_message('START')
    
    mx_module, sym, args, aux = load_network(graph, model)
    image_files, eval_data = load_images(input_dir, batch_size,
        mx_module.data_names[0], data_shape)
    log_info_message('Network output names: {}'.format(sym.list_outputs()))
    
    log_debug_message('START: mx_module.bind(..)')
    mx_module.bind(data_shapes = eval_data.provide_data)
    log_debug_message('FINISH: mx_module.bind(..)')
    
    log_debug_message('START: mx_module.set_params(..)')
    mx_module.set_params(arg_params = args, aux_params = aux)
    log_debug_message('FINISH: mx_module.set_params(..)')
    
    log_debug_message('START: forward(..) for each batch')
    inference_times = []
    outputs = []
    data_iter = iter(eval_data)
    number_iter = (number_iter + batch_size - 1) // batch_size
    
    t0_total = time()
    for i in range(number_iter):
        data_batch = data_iter.next()

        t0 = time()
        mx_module.forward(data_batch, is_train = False)
        for output in mx_module.get_outputs():
            output.wait_to_read()
        t1 = time()
        
        outs = []
        for output in mx_module.get_outputs():
            outs.append(output.asnumpy())
        outputs.append(outs)
    
        inference_times.append(t1 - t0)

    t1_total = time()
    
    total_inference_time = t1_total - t0_total
    log_debug_message('FINISH: forward(..) for each batch')

    save_outputs(need_output, task_type, outputs, image_files,
        output_file, label_file)

    log_debug_message('FINISH')
    return inference_times, total_inference_time


def save_outputs(need_output, task_type, outputs, image_files,
        output_file, label_file):
    log_debug_message('START')
    
    if need_output == True:
        if task_type.lower() == 'classification':
            classification_output(outputs, image_files,
                output_file, label_file)
        else:
            if task_type.lower() == 'detection':
                detection_output(outputs, image_files, output_file, label_file)
            else:
                error_message = 'Incorrect task type \'{}\''.format(task_type)
                log_error_message(error_message)
                raise Exception(error_message)
    
    log_debug_message('FINISH')


def load_labels(label_file):
    log_debug_message('START')

    labels = []
    with open(label_file, 'r') as file:
        labels = [line.rstrip() for line in file]

    log_debug_message('FINISH')
    return labels


def classification_output(outputs, image_files, output_file, label_file):
    log_debug_message('START')

    labels = load_labels(label_file)
    
    file = open(output_file, 'w+')
    for batch_idx in range(len(outputs)):
        probabilities = outputs[batch_idx][0]
        
        batch_size = probabilities.shape[0]
        for image_idx in range(batch_size):
            top5 = np.argsort(probabilities[image_idx])[::-1]
            image = image_files[batch_idx * batch_size + image_idx]
            file.write('{0};{1};{2};{3};{4};{5}\n'.format(image,
                labels[top5[0]], labels[top5[1]],
                labels[top5[2]], labels[top5[3]],
                labels[top5[4]]))
    file.close()
    
    log_debug_message('FINISH')


def detection_output(outputs, image_files, output_file, label_file):
    log_debug_message('START')
    
    labels = load_labels(label_file)
    
    file = open(output_file, 'w+')
    for batch_idx in range(len(outputs)):
        batch_detections = outputs[batch_idx][0]
        
        batch_size = len(batch_detections)
        log_info_message('Batch identifier is {0}, batch size is {1}'.format(
            batch_idx, batch_size))
        for image_idx in range(batch_size):
            image_detections = batch_detections[image_idx]
            
            image_name = image_files[batch_idx * batch_size + image_idx]
            image = mx.image.imread(image_name)
            width = image.shape[0]
            height = image.shape[1]
            log_info_message(
                '\tImage name is \'{0}\', number of detections is {1}'.format(
                        image_name, image_detections.shape[0]))
            for det_idx in range(len(image_detections)):
                (class_id, score, xl, yl, xr, yr) = image_detections[det_idx]
                if (class_id < 0) or (class_id >= len(labels)):
                    # filter false detections (class_id = -1)
                    continue
                xl = int(xl * width)
                yl = int(yl * height)
                xr = int(xr * width)
                yr = int(yr * height)
                log_info_message('\t\t({0}, {1}, {2}, {3}, {4}, {5})'.format(
                    labels[int(class_id)], score, xl, yl, xr, yr))
                file.write('{0};{1};{2};{3};{4};{5};{6}\n'.format(image_name,
                    labels[int(class_id)], score, xl, yl, xr, yr))
        
    file.close()


def three_sigma_rule(time):
    log_debug_message('START')
    
    average_time = np.mean(time)
    sigm = np.std(time)
    upper_bound = average_time + (3 * sigm)
    lower_bound = average_time - (3 * sigm)
    valid_time = []
    for i in range(len(time)):
        if lower_bound <= time[i] <= upper_bound:
            valid_time.append(time[i])
    
    log_debug_message('FINISH')
    return valid_time


def calculate_average_time(time):
    log_debug_message('START')
    
    average_time = np.mean(time)
    
    log_debug_message('FINISH')
    return average_time


def calculate_latency(time):
    log_debug_message('START')
    
    time.sort()
    latency = np.median(time)
    
    log_debug_message('FINISH')
    return latency


def calculate_fps(num_images, time):
    log_debug_message('START')
    
    fps = num_images / time
    
    log_debug_message('FINISH')
    return fps


def create_result_file(file_name):
    log_debug_message('START')
    
    if os.path.isfile(file_name):
        return
    file = open(file_name, 'w')
    head = 'Model;Batch size;Device;IterationCount;Latency;Total time (s);FPS;'
    file.write(head + '\n')
    file.close()
    
    log_debug_message('FINISH')

def write_row(file_name, net_name, batch_size, number_iter,
        latency, total_time, fps):
    log_debug_message('START')
    
    log_info_message('Total time: {0} s'.format(total_time))
    log_info_message('Latency: {0} s'.format(latency))
    log_info_message('FPS: {0}'.format(fps))

    
    row = '{};{};CPU;{};{:.3f};{:.3f};{:.3f}'.format(net_name, batch_size,
        number_iter, latency, total_time, fps)
    file = open(file_name, 'a')
    file.write(row + '\n')
    file.close()
    
    log_debug_message('FINISH')


def main():    
    log.basicConfig(format = '[ %(levelname)s ] %(message)s',
        level = log.DEBUG, stream = sys.stdout)
    
    args = build_argparser().parse_args()
    
    inference_times, total_inference_time = mxnet_benchmark(
        task_type = args.task_type, graph = args.graph,
        model = args.model, input_dir = args.input_dir,
        number_iter = args.number_iter, batch_size = args.batch_size,
        data_shape = { 'nchannels': int(args.data_shape.split(' ')[0]),
                       'width': int(args.data_shape.split(' ')[1]),
                       'height': int(args.data_shape.split(' ')[2]) },
        need_output = args.need_output, label_file = args.label_file,
        output_file = args.output_file)

    inference_times = three_sigma_rule(inference_times)
    latency = calculate_latency(inference_times)
    fps = calculate_fps(args.number_iter, total_inference_time)
    create_result_file(file_name = args.perf_file)
    write_row(args.perf_file, os.path.basename(args.model), args.batch_size,
        args.number_iter, latency, total_inference_time, fps)
 

if __name__ == '__main__':
    main()