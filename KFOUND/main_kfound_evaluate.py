# Copyright 2022 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from model import KFoundModel
from misc import load_config
from datasets.datasets import build_dataset
from evaluation.saliency import evaluate_saliency
from evaluation.uod import evaluation_unsupervised_object_discovery
from KANLinear import KAN_Convolutional_Layer,Kan_and_Conv;
import torch.nn as nn;
import torch;

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'Evaluation of FOUND',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUTS-TEST" --eval-type saliency --evaluation-mode single
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "DUTS-TEST" --eval-type saliency --evaluation-mode single
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUTS-TEST" --eval-type saliency --evaluation-mode multi
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "DUTS-TEST" --eval-type saliency --evaluation-mode multi
    
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode single
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode single
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode multi
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode multi
    
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "ECSSD" --eval-type saliency --evaluation-mode single
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "ECSSD" --eval-type saliency --evaluation-mode single
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "ECSSD" --eval-type saliency --evaluation-mode multi
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "ECSSD" --eval-type saliency --evaluation-mode multi
    
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUTS-TEST" --eval-type saliency --evaluation-mode single --g 9 --model-weights ./outputs/g9/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt 
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUTS-TEST" --eval-type saliency --evaluation-mode multi --g 9 --model-weights ./outputs/g9/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt 
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode single --g 9 --model-weights ./outputs/g9/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt 
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode multi --g 9 --model-weights ./outputs/g9/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt 
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "ECSSD" --eval-type saliency --evaluation-mode single --g 9 --model-weights ./outputs/g9/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter400.pt 
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "ECSSD" --eval-type saliency --evaluation-mode multi --g 9 --model-weights ./outputs/g9/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter400.pt 
    
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUTS-TEST" --eval-type saliency --evaluation-mode single --g 11 --model-weights ./outputs/g11/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt 
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUTS-TEST" --eval-type saliency --evaluation-mode multi --g 11 --model-weights ./outputs/g11/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt 
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode single --g 11 --model-weights ./outputs/g11/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt 
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode multi --g 11 --model-weights ./outputs/g11/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt 
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "ECSSD" --eval-type saliency --evaluation-mode single --g 11 --model-weights ./outputs/g11/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter200.pt 
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "ECSSD" --eval-type saliency --evaluation-mode multi --g 11 --model-weights ./outputs/g11/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter200.pt 
    
    
    
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode single --model-weights ./outputs/full_size/crf/Conv/FOUND-DUTS-TR/decoder_weights_niter500.pt
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode multi --model-weights ./outputs/full_size/crf/Conv/FOUND-DUTS-TR/decoder_weights_niter500.pt
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "DUTS-TEST" --eval-type saliency --evaluation-mode single --model-weights ./outputs/full_size/crf/Conv/FOUND-DUTS-TR/decoder_weights_niter500.pt
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "DUTS-TEST" --eval-type saliency --evaluation-mode multi --model-weights ./outputs/full_size/crf/Conv/FOUND-DUTS-TR/decoder_weights_niter500.pt
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "ECSSD" --eval-type saliency --evaluation-mode single --model-weights ./outputs/full_size/crf/Conv/FOUND-DUTS-TR/decoder_weights_niter500.pt
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "ECSSD" --eval-type saliency --evaluation-mode multi --model-weights ./outputs/full_size/crf/Conv/FOUND-DUTS-TR/decoder_weights_niter500.pt
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode single --model-weights ./outputs/latest/Conv/FOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode multi --model-weights ./outputs/latest/Conv/FOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "DUTS-TEST" --eval-type saliency --evaluation-mode single --model-weights ./outputs/latest/Conv/FOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "DUTS-TEST" --eval-type saliency --evaluation-mode multi --model-weights ./outputs/latest/Conv/FOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "ECSSD" --eval-type saliency --evaluation-mode single --model-weights ./outputs/latest/Conv/FOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt
    # python3 ./main_found_evaluate.py --decoder Conv --dataset-dir ../data/ --dataset-eval "ECSSD" --eval-type saliency --evaluation-mode multi --model-weights ./outputs/latest/Conv/FOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode single --model-weights ./outputs/full_size/crf/Kan/KFOUND-DUTS-TR-11/decoder_weights_niter500.pt --g 11
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode multi --model-weights ./outputs/full_size/crf/Kan/KFOUND-DUTS-TR-11/decoder_weights_niter500.pt --g 11
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode single --model-weights ./outputs/full_size/crf/Kan/KFOUND-DUTS-TR-13/decoder_weights_niter400.pt --g 13
    # python3 ./main_found_evaluate.py --decoder Kan --dataset-dir ../data/ --dataset-eval "DUT-OMRON" --eval-type saliency --evaluation-mode multi --model-weights ./outputs/full_size/crf/Kan/KFOUND-DUTS-TR-13/decoder_weights_niter400.pt --g 13
    
    parser.add_argument(
        "--eval-type",
        type=str,
        choices=["saliency", "uod"],
        help="Evaluation type."
    )
    parser.add_argument(
        "--dataset-eval",
        type=str,
        choices=["ECSSD", "DUT-OMRON", "DUTS-TEST", "VOC07", "VOC12", "COCO20k"],
        help="Name of evaluation dataset."
    )
    parser.add_argument(
        "--dataset-set-eval",
        type=str,
        default=None,
        help="Set of the dataset."
    )
    parser.add_argument(
        "--apply-bilateral",
        action="store_true", 
        help="use bilateral solver."
    )
    parser.add_argument(
        '--d',
        type=int,
        default=3,
        help="degree of the bspline curve"
    );
    
    parser.add_argument(
        '--s',
        type=str,
        default='dcrf',
        choices=["dcrf", "bs"],
        help='smoothing technique'
    );
    
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        default="multi",
        choices=["single", "multi"],
        help="Type of evaluation."
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kfound_DUTS-TR.yaml",
    )
    
    
    parser.add_argument(
        "--decoder",
        type=str,
        default="Kan",
        choices=["Kan", "Conv","KanConv"],
        help = "Select between either Kan or Conv"
    )
    
    parser.add_argument(
        '--g',
        type=int,
        default=5,
        help='KAN Grid size'
    )
    
    args = parser.parse_args()
    print(args.__dict__,args)

    # Configuration
    config = load_config(args.config)

    # ------------------------------------
    # Load the model
    model = KFoundModel(vit_model=config.model["pre_training"],
                        vit_arch=config.model["arch"],
                        vit_patch_size=config.model["patch_size"],
                        enc_type_feats=config.kfound["feats"],
                        bkg_type_feats=config.kfound["feats"],
                        bkg_th=config.kfound["bkg_th"])
    if args.decoder == "Kan":
        model.decoder = KAN_Convolutional_Layer(
                        in_channels=384,  # Number of input channels
                        out_channels=1,  # Number of output channels to match 1x1 convolution
                        grid_size=args.g,
                        spline_order=args.d,
                        scale_noise=0.1,
                        scale_base=1.0,
                        scale_spline=1.0,
                        base_activation=nn.SiLU,
                        grid_eps=0.02,
                        grid_range=[-1, 1]
                    )
    elif args.decoder == "KanConv":
        model.decoder = Kan_and_Conv(
            in_channels=384,
            out_channels=1,
            grid_size=args.g,
            spline_order=args.d
        );
        
        if args.model_weights == None:
            args.model_weights = "./outputs/v2/Kan/KFOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt";
    elif args.model_weights == None:
        args.model_weights = "./outputs/Conv/FOUND-DUTS-TR-vit_small8/decoder_weights_niter1000.pt"
        
    # Load weights
    model.decoder_load_weights(args.model_weights)
    model.eval()
    print(f"Model {args.model_weights} loaded correctly.")

    # ------------------------------------
    # Build the validation set
    
    val_dataset = build_dataset(
        root_dir=args.dataset_dir,
        dataset_name=args.dataset_eval,
        dataset_set=args.dataset_set_eval,
        for_eval=True,
        evaluation_type=args.eval_type,
    )
    print(f"\nBuilding dataset {val_dataset.name} (#{len(val_dataset)} images)")
    
    # ------------------------------------
    # Training
    print(f"\nStarted evaluation on {val_dataset.name}")
    if args.eval_type == "saliency":
        # evaluate_saliency(
        #     val_dataset,
        #     model=model,
        #     evaluation_mode=args.evaluation_mode,
        #     apply_bilateral=args.apply_bilateral,
        # )
        evaluate_saliency(
            dataset=val_dataset,
            model=model,
            writer=None,
            batch_size=1,
            n_iter=0,  # Start iteration
            apply_bilateral=True,
            im_fullsize=True,
            method="pred",  # Default method
            apply_weights=True,
            evaluation_mode=args.evaluation_mode,
            s = args.s
        )
        torch.cuda.empty_cache();
    elif args.eval_type == "uod":
        if args.apply_bilateral:
            raise ValueError("Not implemented.")

        evaluation_unsupervised_object_discovery(
            val_dataset,
            model=model,
            evaluation_mode=args.evaluation_mode,
        )
    else:
        raise ValueError("Other evaluation method to come.")
    
    
