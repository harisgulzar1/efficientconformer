# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pytorch Pytorch
import torch

# Functions and Utils
from functions import *
from utils.preprocessing import *
import torch.nn as nn

# Other
import json
import argparse
import os
from thop import profile
from torchsummary import summary
import copy
import numpy as np
from models.model_ctc import ModelCTC

class QuantizedConf(ModelCTC):
    def __init__(self, encoder_params, tokenizer_params, training_params, decoding_params, name):
        super(QuantizedConf, self).__init__(encoder_params, tokenizer_params, training_params, decoding_params, name)
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        # self.model_fp32 = name

    def forward(self, batch):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model

        x, _, x_len, _ = batch
        # print("0", x.type())
        x = self.quant(x)
        # print("BEGIN", x.type())
        logits, logits_len, attentions = self.encoder(x, x_len)
        
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        # Tokenizer
        # print(logits)
        x = self.fc(logits)
        # print(x)
        x = self.dequant(x)
        # print("FINAL: ", x.type())
        

        return x, logits_len, attentions



    def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1, 12345, 80, 12345)):

        model_1.to(device)
        model_2.to(device)

        for _ in range(num_tests):
            x = torch.rand(size=input_size).to(device)
            y1 = model_1(x).detach().cpu().numpy()
            y2 = model_2(x).detach().cpu().numpy()
            if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
                print("Model equivalence test sample failed: ")
                print(y1)
                print(y2)
                return False

        return True

def nested_children(m):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output

def main(rank, args):

    # Process rank
    args.rank = rank

    # Distributed Computing
    if args.distributed:
        torch.cuda.set_device(args.rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)

    # Load Config
    with open(args.config_file) as json_config:
        config = json.load(json_config)

    # Device
    # device = torch.device("cuda:0,1")
    device = torch.device("cuda:" + str(args.rank) if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    # Create Tokenizer
    if args.create_tokenizer:

        if args.rank == 0:
            print("Creating Tokenizer")
            create_tokenizer(config["training_params"], config["tokenizer_params"])

        if args.distributed:
            torch.distributed.barrier()

    # Create Model
    model = create_model(config).to(device)
    quantized_model = QuantizedConf(
        encoder_params=config["encoder_params"],
        tokenizer_params=config["tokenizer_params"],
        training_params=config["training_params"],
        decoding_params=config["decoding_params"],
        name=config["model_name"])    

    # Load Model
    if args.initial_epoch is not None:

        pre_trained = torch.load(config["training_params"]["callback_path"] + "checkpoints_" + str(args.initial_epoch) + ".ckpt")
        new=list(pre_trained['model_state_dict'].items())

        my_model_kvpair=model.state_dict()
        count=0
        for key,value in my_model_kvpair.items():
            print(key)
            layer_name,weights=new[count]      
            my_model_kvpair[key]=weights
            count+=1
            print(count)

        model.load_state_dict(my_model_kvpair)
        
        # model.load(config["training_params"]["callback_path"] + "checkpoints_" + str(args.initial_epoch) + ".ckpt")
        print("Model Loaded!")
    else:
        args.initial_epoch = 0

    # Load Encoder Only
    if args.initial_epoch_encoder is not None:
        # model.load_encoder(config["training_params"]["callback_path_encoder"] + "checkpoints_" + str(args.initial_epoch_encoder) + ".ckpt")
        pre_trained = torch.load(config["training_params"]["callback_path"] + "checkpoints_" + str(args.initial_epoch_encoder) + ".ckpt")
        new=list(pre_trained['model_state_dict'].items())

        my_model_kvpair= quantized_model.encoder.state_dict()
        
        print(len(my_model_kvpair.items()))
        # print(len(new))
        # for key, value in my_model_kvpair.items():
        #     print(key)
        count=0
        for key,value in my_model_kvpair.items():
            layer_name,weights=new[count]   
            print(key, "\t:\t", layer_name)
            my_model_kvpair[key]=weights
            count+=1
            # print(count)

        quantized_model.encoder.load_state_dict(my_model_kvpair)
        
        # model.load(config["training_params"]["callback_path"] + "checkpoints_" + str(args.initial_epoch) + ".ckpt")
        print("Model Loaded!")

    # Load LM
    if args.initial_epoch_lm:

        # Load LM Config
        with open(config["decoding_params"]["lm_config"]) as json_config:
            config_lm = json.load(json_config)

        # Create LM
        model.lm = create_model(config_lm).to(device)

        # Load LM
        model.lm.load(config_lm["training_params"]["callback_path"] + "checkpoints_" + str(args.initial_epoch_lm) + ".ckpt")

    # Model Summary
    if args.rank == 0:
        model.summary(show_dict=args.show_dict)

    # Distribute Strategy
    if args.distributed:
        if args.rank == 0:
            print("Parallelize model on", args.world_size, "GPUs")
        model.distribute_strategy(args.rank)

    # Parallel Strategy
    if args.parallel and not args.distributed:
        print("Parallelize model on", torch.cuda.device_count(), "GPUs")
        model.parallel_strategy()

    # Prepare Dataset
    if args.prepare_dataset:

        if args.rank == 0:
            print("Preparing dataset")
            prepare_dataset(config["training_params"], config["tokenizer_params"], model.tokenizer)

        if args.distributed:
            torch.distributed.barrier()

    # Load Dataset
    dataset_train, dataset_val = load_datasets(config["training_params"], config["tokenizer_params"], args)

    ###############################################################################
    # Modes
    ###############################################################################

    # Stochastic Weight Averaging
    if args.swa:

        model.swa(dataset_train, callback_path=config["training_params"]["callback_path"], start_epoch=args.swa_epochs[0] if args.swa_epochs else None, end_epoch=args.swa_epochs[1] if args.swa_epochs else None, epochs_list=args.swa_epochs_list, update_steps=args.steps_per_epoch, swa_type=args.swa_type)

    # Training


    elif args.mode.split("-")[0] == "training":

    ###########################################
        # Using un-fused model will fail.
        # Because there is no quantized layer implementation for a single batch normalization layer.
        # quantized_model = QuantizedConf(model_fp32=model)
        # Select quantization schemes from
        # https://pytorch.org/docs/stable/quantization-support.html

        print(quantized_model)
        quantized_model.train()

        print("FUSION!")
        # quantized_model = torch.quantization.fuse_modules(quantized_model, [["encoder.subsampling_module.conv1","encoder.subsampling_module.bn1", "encoder.subsampling_module.relu1"]], inplace=True)
        # quantized_model = torch.quantization.fuse_modules(quantized_model, [["encoder.conformerb.convolution_module.conv3", "encoder.conformerb.convolution_module.glu2"]], inplace=True)
       
       #for i in range(14):
            #quantized_model = torch.quantization.fuse_modules(quantized_model, [[f"encoder.blocks.{i}.convolution_module.layers.2", f"encoder.blocks.{i}.convolution_module.layers.3"]], inplace=True)
            #quantized_model = torch.quantization.fuse_modules(quantized_model, [[f"encoder.blocks.{i}.convolution_module.layers.4", f"encoder.blocks.{i}.convolution_module.layers.5"]], inplace=True)
        # for module_name, module in fused_model.named_children():
        #     if "layer" in module_name:
        #         for basic_block_name, basic_block in module.named_children():
        #             torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
        #             for sub_block_name, sub_block in basic_block.named_children():
        #                 if sub_block_name == "downsample":
        #                     torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)


        quantconfig = torch.quantization.get_default_qconfig("fbgemm")
        print(quantconfig)
        
        quantized_model.encoder.subsampling_module.conv1.qconfig = quantconfig
        # quantized_model.qconfig = quantconfig
        # quantized_model.encoder.conformerb.convolution_module.qconfig = quantconfig
        # quantized_model.encoder.linear.qconfig = None
        # quantized_model.encoder.conformerb.multi_head_self_attention_module.mhsa.qconfig = None
        # quantized_model.encoder.conformerb.convolution_module.conv1.qconfig = None
        # quantized_model.encoder.conformerb.convolution_module.conv1.qconfig = None
        # quantized_model.encoder.conformerb.convolution_module.conv2.qconfig = None
        # quantized_model.encoder.conformerb.convolution_module.conv3.qconfig = None


        # quantized_model.fc.qconfig = quantconfig

        # quantized_model.encoder.preprocessing.qconfig = None
        # quantized_model.encoder.augment.qconfig = None

        torch.quantization.prepare_qat(quantized_model, inplace=True)
        # torch.quantization.prepare_qat(quantized_model.encoder.subsampling_module.bn1, inplace=True)
        # torch.quantization.prepare_qat(quantized_model.encoder.subsampling_module.relu1, inplace=True)

        #######################################
        # quantized_model_prepared.train()
        print(quantized_model)
        quantized_model.to(device)

        quantized_model.fit(dataset_train, 
            config["training_params"]["epochs"], 
            dataset_val=dataset_val, 
            val_steps=args.val_steps, 
            verbose_val=args.verbose_val, 
            initial_epoch=int(args.initial_epoch), 
            callback_path=config["training_params"]["callback_path"], 
            steps_per_epoch=args.steps_per_epoch,
            mixed_precision=config["training_params"]["mixed_precision"],
            accumulated_steps=config["training_params"]["accumulated_steps"],
            saving_period=args.saving_period,
            val_period=args.val_period)
        
        quantized_model.to('cpu')

        model_int8 = torch.quantization.convert(quantized_model, inplace=True)
        print(model_int8)
        model_int8.save("callbacks/EfficientConformerCTCSmall/checkpoints_q.ckpt")
        print("Gready Search Evaluation")
        wer, _, _, _ = model_int8.evaluate(dataset_val, eval_steps=args.val_steps, verbose=args.verbose_val, beam_size=1, eval_loss=args.eval_loss)
        if args.rank == 0:
            print("Geady Search WER : {:.2f}%".format(100 * wer))
        
        # quantized_model.eval()
        # quantized_model.to('cpu')
        
        # Custom quantization configurations
        # quantization_config = torch.quantization.default_qconfig
        # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
        # quantized_model.encoder.qconfig = quantization_config
        # torch.backends.quantized.engine = 'fbgemm'
        # # Print quantization configurations
        #print(quantized_model.qconfig)

        # https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare_qat

        # torch.quantization.prepare_qat(quantized_model.fc, inplace=True)
        #torch.quantization.prepare_qat(quantized_model., inplace=True)
        # print(quantized_model.fc)

        # # Use training data for calibration.
        # print("Training QAT Model...")
        # quantized_model.train()
        # quantized_model.to(device)
        #print(quantized_model.fc)


        # Using high-level static quantization wrapper
        # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
        # quantized_model = torch.quantization.quantize_qat(model=quantized_model, run_fn=train_model, run_args=[train_loader, test_loader, cuda_device], mapping=None, inplace=False)

        #quantized_model = torch.quantization.convert(quantized_model, inplace=True)

        # Print quantized model.

        # model_int8.eval()
        # torch.save(model_int8.state_dict())
        # model=model_int8

    # Evaluation
    elif args.mode.split("-")[0] == "validation" or args.mode.split("-")[0] == "test":

        # Gready Search Evaluation
        if args.gready or model.beam_size is None:

            # flops, params = profile(model, inputs=(dataset_val, ))

            # print("FLOPS: ", flops)
            # print("PARAMs: ", params)
            if args.rank == 0:
                print("Gready Search Evaluation")
            wer, _, _, _ = model.evaluate(dataset_val, eval_steps=args.val_steps, verbose=args.verbose_val, beam_size=1, eval_loss=args.eval_loss)
            
            if args.rank == 0:
                print("Geady Search WER : {:.2f}%".format(100 * wer))
        
        # Beam Search Evaluation
        else:

            if args.rank == 0:
                print("Beam Search Evaluation")
            wer, _, _, _ = model.evaluate(dataset_val, eval_steps=args.val_steps, verbose=args.verbose_val, beam_size=model.beam_size, eval_loss=False)
            
            if args.rank == 0:
                print("Beam Search WER : {:.2f}%".format(100 * wer))
    
    # Eval Time
    elif args.mode.split("-")[0] == "eval_time":

        print("Model Eval Time")
        inf_time = model.eval_time(dataset_val, eval_steps=args.val_steps, beam_size=1, rnnt_max_consec_dec_steps=args.rnnt_max_consec_dec_steps, profiler=args.profiler)
        print("eval time : {:.2f}s".format(inf_time))

    elif args.mode.split("-")[0] == "eval_time_encoder":

        print("Encoder Eval Time")
        enc_time = model.eval_time_encoder(dataset_val, eval_steps=args.val_steps, profiler=args.profiler)
        print("eval time : {:.2f}s".format(enc_time))

    elif args.mode.split("-")[0] == "eval_time_decoder":

        print("Decoder Eval Time")
        dec_time = model.eval_time_decoder(dataset_val, eval_steps=args.val_steps, profiler=args.profiler)
        print("eval time : {:.2f}s".format(dec_time))
    
    # Destroy Process Group
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",          type=str,   default="configs/EfficientConformerCTCSmall.json",  help="Json configuration file containing model hyperparameters")
    parser.add_argument("-m", "--mode",                 type=str,   default="training",                                 help="Mode : training, validation-clean, test-clean, eval_time, audio_lat, ...")
    parser.add_argument("-d", "--distributed",          type=str,   default=None,                                       help="Distributed data parallelization")
    parser.add_argument("-i", "--initial_epoch",        type=str,   default= None,                                       help="Load model from checkpoint")
    parser.add_argument("-lm", "--initial_epoch_lm",    type=str,   default=None,                                       help="Load language model from checkpoint")
    parser.add_argument("--initial_epoch_encoder",      type=str,   default= "1",                                       help="Load model encoder from encoder checkpoint")
    parser.add_argument("-p", "--prepare_dataset",      action="store_true",                                            help="Prepare dataset for training")
    parser.add_argument("-j", "--num_workers",          type=int,   default=0,                                          help="Number of data loading workers")
    parser.add_argument("--create_tokenizer",           action="store_true",                                            help="Create model tokenizer")
    parser.add_argument("--batch_size_eval",            type=int,   default=8,                                          help="Evaluation batch size")
    parser.add_argument("--verbose_val",                action="store_true",                                            help="Evaluation verbose")
    parser.add_argument("--val_steps",                  type=int,   default=None,                                       help="Number of validation steps")
    parser.add_argument("--steps_per_epoch",            type=int,   default=None,                                       help="Number of steps per epoch")
    parser.add_argument("--world_size",                 type=int,   default=torch.cuda.device_count(),                  help="Number of available GPUs")
    parser.add_argument("--cpu",                        action="store_true",                                            help="Load model on cpu")
    parser.add_argument("--show_dict",                  action="store_true",                                            help="Show model dict summary")
    parser.add_argument("--swa",                        action="store_true",                                            help="Stochastic weight averaging")
    parser.add_argument("--swa_epochs",                 nargs="+",  default=None,                                       help="Start epoch / end epoch for swa")
    parser.add_argument("--swa_epochs_list",            nargs="+",  default=None,                                       help="List of checkpoints epochs for swa")
    parser.add_argument("--swa_type",                   type=str,   default="equal",                                    help="Stochastic weight averaging type (equal/exp)")
    parser.add_argument("--parallel",                   action="store_true",                                            help="Parallelize model using data parallelization")
    parser.add_argument("--rnnt_max_consec_dec_steps",  type=int,   default=None,                                       help="Number of maximum consecutive transducer decoder steps during inference")
    parser.add_argument("--eval_loss",                  action="store_true",                                            help="Compute evaluation loss during evaluation")
    parser.add_argument("--gready",                     action="store_true",                                            help="Proceed to a gready search evaluation")
    parser.add_argument("--saving_period",              type=int,   default=1,                                          help="Model saving every 'n' epochs")
    parser.add_argument("--val_period",                 type=int,   default=1,                                          help="Model validation every 'n' epochs")
    parser.add_argument("--profiler",                   action="store_true",                                            help="Enable eval time profiler")

    # Parse Args
    args = parser.parse_args()

    # Run main
    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        torch.multiprocessing.spawn(main, nprocs=args.world_size, args=(args,))  
    else:
        main(3, args)
