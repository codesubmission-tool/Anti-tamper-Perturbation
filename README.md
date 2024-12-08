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

The links for datasets can be found from [FFHQ](https://github.com/NVlabs/ffhq-dataset), [CelebA-HQ](https://github.com/VinAIResearch/Anti-DreamBooth/tree/main), [VGGFace2](https://github.com/VinAIResearch/Anti-DreamBooth/tree/main).

## :zap: Quick Inference

**Authorized Perturbation**

Modify the settings defined on ./configs/authorization.yaml

```
cd authorization

bash scripts/infer.sh 
```
**Protection Perturbation**

Modify the settings defined on ./configs/protection.yaml

```
cd protection

python protection --method CAAT
```

**Verification after the Perturbation**

Modify the settings defined on ./configs/verification.yaml

```
cd authorization

bash scripts/verify.sh 
```

**Generation after the Perturbation**

Modify the settings defined on ./configseval_{CelebA-HQ}/{VGGFac2}

```
cd evaluation

python generate.py --dataset CelebA-HQ --method CAAT
```

**Protection Performance calculation after the Generation**

Download ID embedding from [release sources](https://github.com/codesubmission-tool/Anti-tamper-Perturbation/releases).  

Modify the settings defined on ./configs/metrics.yaml

Download [LIQE.pt](https://drive.google.com/file/d/1GoKwUKNR-rvX11QbKRN8MuBZw2hXKHGh/view) from [url](https://github.com/zwx8981/LIQE). Place it to ./metrics/LIQE/checkpoints

```
cd metrics

python eval.py --dataset CelebA-HQ --method CAAT

python show.py --path evaluation_results
```

<!-- **Complete ATP Pipeline** -->

<!-- ## :computer: Training -->

