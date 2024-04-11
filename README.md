# BoMD: Bag of Multi-label Descriptors for Noisy Chest X-ray Classification
Official code for "BoMD: Bag of Multi-label Descriptors for Noisy Chest X-ray Classification"


> **BoMD: Bag of Multi-label Descriptors for Noisy Chest X-ray Classification**,<br />
> [Yuanhong Chen*](https://scholar.google.com/citations?user=PiWKAx0AAAAJ&hl=en&oi=ao), [Fengbei Liu*](https://fbladl.github.io/), [Hu Wang](https://huwang01.github.io/), [Chong Wang](https://scholar.google.com/citations?user=IWcTej4AAAAJ&hl=en&oi=ao), [Yu Tian](https://yutianyt.com/), [Yuyuan Liu](https://scholar.google.com/citations?user=SibDXFQAAAAJ&hl=zh-CN), [Gustavo Carneiro](https://www.surrey.ac.uk/people/gustavo-carneiro).            
> *ICCV 2023 ([arXiv 2203.01937](https://arxiv.org/abs/2203.01937))*



## Download links

### Weights
* [Word embeddings](https://drive.google.com/drive/folders/1S3kL6KGtom_LTsqivdrlbdz_yXYKH8aE?usp=sharing) extracted by the BLUEBert model.
* [Classifier weights trained on NIH](https://drive.google.com/drive/folders/1KzMOSRP_Q121f1ikwyIHn1FMSEDXUiO2?usp=sharing)
* [MID \& NSD weights for run-20220816_175557-6dn2k0id](https://drive.google.com/drive/folders/1NeafemhJg0HGf52Fzg9kLmhkyKMEhXeQ?usp=drive_link)

### Datasets
Download the raw data:
[ChestXpert](https://www.kaggle.com/datasets/willarevalo/chexpert-v10-small), 
[NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC), 
[PadChest](http://bimcv.cipf.es/bimcv-projects/padchest/), 
[OpenI](https://openi.nlm.nih.gov/faq), 
[NIH-GOOGLE](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest#additional_labels)


Download the processed [OpenI & PadChest](https://drive.google.com/drive/folders/1tSNMHxhs98AFctTOur9MTMLhYfiKEJ_-?usp=sharing)

## Training
Train BoMD on NIH with the following command:
```
bash run_train_bomd_nih.sh
```
Train classificaiton head on NIH with the MID and NSD checkpoints:
```
bash run_train_bomd_nih_cls
```
## Testing
```
bash run_eval_bomd_nih.sh
```


## Reference
```bibtex
@article{chen2022bomd,
  title={BoMD: Bag of Multi-label Descriptors for Noisy Chest X-ray Classification},
  author={Chen, Yuanhong and Liu, Fengbei and Wang, Hu and Wang, Chong and Tian, Yu and Liu, Yuyuan and Carneiro, Gustavo},
  journal={arXiv preprint arXiv:2203.01937},
  year={2022}
}
```
