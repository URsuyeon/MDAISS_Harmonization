# 사용법:
#   python -m scripts.run_stargan3d           # 학습만 실행 (기본)
#   python -m scripts.run_stargan3d --train   # 학습만 실행
#   python -m scripts.run_stargan3d --transform   # 변환만 실행
#   python -m scripts.run_stargan3d --train --transform   # 학습 후 변환 모두 실행

import os
import torch
from pathlib import Path
import yaml
import argparse

# StarGAN3Dv2 관련 모듈 import
from models.stargan3d_v2 import StarGAN3Dv2,  MedicalVolumeDataset

def train_stargan(config, stargan, data_dir, output_dir):
    input_shape = tuple(config["data"]["input_shape"])
    batch_size = config["data"]["batch_size"]
    num_epochs = config["train"]["epochs"]

    stargan.train(
        data_dir=data_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        generator_lr=config["train"]["generator_lr"],
        discriminator_lr=config["train"]["discriminator_lr"],
        beta1=config["train"]["beta1"],
        beta2=config["train"]["beta2"],
        lambda_cyc_max=config["train"]["lambda_cyc_max"],
        lambda_sty_max=config["train"]["lambda_sty_max"],
        lambda_con_max=config["train"]["lambda_con_max"],
        lambda_cls=config["train"]["lambda_cls"],
        lambda_grl_max=config["train"]["lambda_grl_max"],
        warmup_cyc_epochs=config["train"]["warmup_cyc_epochs"],
        warmup_sty_epochs=config["train"]["warmup_sty_epochs"],
        warmup_con_epochs=config["train"]["warmup_con_epochs"],
        warmup_grl_epochs=config["train"]["warmup_grl_epochs"],
        two_stage_training=config["train"].get("two_stage_training", True),
        save_interval=config["train"]["save_interval"],
        grad_accum_steps=config["train"]["grad_accum_steps"],
        use_mixed_precision=config["train"]["use_mixed_precision"],
        use_warmup=config["train"].get("use_warmup", True),
        use_wandb=config.get("logging", {}).get("use_wandb", False) 
    )

def transform_stargan(config, stargan, data_dir, output_dir):
    # 모든 split에 대해 변환 실행
    for split in ["test", "train", "val"]:
        stargan.transform_dataset(
            data_dir=data_dir,
            output_dir=output_dir,
            split=split
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='학습 실행')
    parser.add_argument('--transform', action='store_true', help='이미지 변환 실행')
    args = parser.parse_args()

    if not args.train and not args.transform:
        args.train = True

    # stargan_config.yaml 로드
    config_path = Path(__file__).parent / "config_stargan.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = config["data"]["root"]
    base_model_dir = config["project"]["save_dir"]
    output_dir = os.path.join(data_dir, "transformed", config["project"]["name"])
    input_shape = tuple(config["data"]["input_shape"])
    style_dim = config["model"]["style_dim"]
    content_dim = config["model"]["content_dim"]

    temp_dataset = MedicalVolumeDataset(csv_file=os.path.join(data_dir, "train_dataset.csv"))
    num_domains = temp_dataset.num_domains
    del temp_dataset

    # 주요 설정값을 폴더명에 포함
    training_folder_name = (
        f"batch_{config['data']['batch_size']}_glr_{config['train']['generator_lr']}_dlr_{config['train']['discriminator_lr']}_epochs_{config['train']['epochs']}"
        f"_cyc{config['train']['lambda_cyc_max']}_sty{config['train']['lambda_sty_max']}_con{config['train']['lambda_con_max']}_cls{config['train']['lambda_cls']}_grl{config['train']['lambda_grl_max']}"
        f"_warmup{int(config['train'].get('use_warmup', True))}_twostage{int(config['train'].get('two_stage_training', True))}"
    )

    if args.transform and not args.train:
        model_dir = os.path.join(base_model_dir, training_folder_name)
    else:
        model_dir = base_model_dir

    stargan = StarGAN3Dv2(
        model_dir=model_dir,
        input_shape=input_shape,
        num_domains=num_domains, 
        style_dim=style_dim,
        content_dim=content_dim
    )

    if args.train:
        train_stargan(config, stargan, data_dir, output_dir)
    if args.transform:
        transform_stargan(config, stargan, data_dir, output_dir)

if __name__ == "__main__":
    main()
