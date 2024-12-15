# :book: Anti-Tamper Protection for Unauthorized Individual Image Generation



<div style="text-align: center;">
    <img src="./assets/concept.jpg" alt="Page 1 of PDF" width="800" />
</div>

## <a name="installation"></a>:crossed_swords: Installation

```shell
pip install -r requirements.txt
```

## <a name="pretrained_models"></a>:dna: Pretrained Models And Datasets

The pretrained model weights and the random mask can be found from [release sources](https://github.com/codesubmission-tool/Anti-tamper-Perturbation/releases) of this repo.

The links for datasets can be found from [FFHQ](https://github.com/NVlabs/ffhq-dataset), [CelebA-HQ](https://drive.google.com/drive/folders/1JX4IM6VMkkv4rER99atS4x4VGnoRNByV), [VGGFace2](https://drive.google.com/drive/folders/1JX4IM6VMkkv4rER99atS4x4VGnoRNByV).

## :zap: Quick Inference

**Authorized Perturbation**

Modify the settings defined on ./configs/authorization.yaml. You need to set down the data path and dataset you use and where to store the perturbed images.

```
cd authorization

bash scripts/infer.sh 
```
Inside infer.sh, you can define the path to the pretrained authoirzation model.

**Protection Perturbation**

Modify the settings defined on ./configs/protection.yaml. You need to define the path to the authorized image, the output path and the path to the mask parameters.

```
cd protection

python protect.py --method CAAT
```
you can define the protection perturbation algorithm by change hyper-parameter --method to ['CAAT','ANTIDB', 'ADVDM' ,'METACLOAK']

**Verification after the Protection Perturbation**

Modify the settings defined on ./configs/verification.yaml. You need to define the path to the authorized image (message_dir), the output path, the method used for protection and the dataset name.

```
cd authorization

bash scripts/verify.sh 
```
Inside verify.sh, you can define the path to the pretrained authoirzation model.

**Generation after the Perturbation**

Modify the settings defined on ./configseval_{CelebA-HQ}/{VGGFac2}. You need to define the path to original dataset images and the output path of the generation.

```
cd evaluation

python generate.py --dataset CelebA-HQ --method CAAT
```
You need to define the dataset name and protection perturbation name by --dataset and --method here. 

**Protection Performance calculation after the Generation**

Download ID embedding from [release sources](https://github.com/codesubmission-tool/Anti-tamper-Perturbation/releases).  

Modify the settings defined on ./configs/metrics.yaml. You need to define the path to the image to be evaluated and the path to store **result record file**, and the prompt used to do generation (for metirc "ImageReward" calculation).

Download [LIQE.pt](https://drive.google.com/file/d/1GoKwUKNR-rvX11QbKRN8MuBZw2hXKHGh/view) from [url](https://github.com/zwx8981/LIQE). Place it to ./metrics/LIQE/checkpoints

```
cd metrics

python eval.py --dataset CelebA-HQ --method CAAT

python show.py --path evaluation_results
```
You need to define the dataset name and protection perturbation name by --dataset and --method here. 
The --path is used to define the path to **result record file**. By modifying the eval.py code in line 63 to line 64 (commented), you can also evaluate the purified generation result.

## :smiling_imp: Generation after the Purifications

**Naive Purification**

For the naive purifications, the purification process is nested into the dreambooth generation process.

change the code in line 36 of generate.py
```
bash scripts/train_DB.sh
```
to
```
bash scripts/train_DB_withPurified.sh
```

For the SOTA purification GridPure, we need to purify the protected image first and then do generation.

**SOTA Purification**
```
cd evaluation/GrIDPure
bash purify_by_gridpure.sh input_path output_path
```

**Generation**

Modify the settings defined on ./configseval_{CelebA-HQ}/{VGGFac2}. You need to define the path to original dataset images and the output path of the generation.
```
cd evaluation

python generate.py --dataset CelebA-HQ --method CAAT
```

## :unlock: Verification after the Purifications

Modify the settings defined on ./configs/verification.yaml. You need to define the path to the authorized image (message_dir), the output path, the method used for protection and the dataset name.
```
cd authorization

bash scripts/verify_record.sh 
```
Inside verify_record.sh, you can define the path to the pretrained authoirzation model.

the bit error values are stored in metrics/results by default and can be used to calculate Protection Success Rate (PSR).


## :lock: Protection Success Rate Calculation

- Modify the settings in merics/PSR.py defined in line 18, 22, 23
- Modify the settings in merics/PSR_wauth.py defined in line 20, 27, 28, 29

PSR.py is for the images only with protection perturbation, while PSR_wauth.py is for the images with authorization perturbation and protection perturbation (i.e., ATP).
```
cd metrics

python PSR.py

python PSR_wauth.py
```


## :computer: Training

You can also train your own authorization model.

Modify the settings in ./configs/authorization.yaml

```
cd authorization

bash scripts/train.sh 
```
<!-- 
## Acknowledgement -->


