# data_auditing_false_claim

## Environment
```
python==3.9.15
datasets==2.16.1
numpy==1.23.5
peft==0.10.0
pillow==9.3.0
scikit-learn==1.1.2
scipy==1.8.0
tokenizers==0.15.2
torch==2.1.2
torchvision==0.16.2
transformers==4.35.2
bitsandbytes==0.43.2
open-clip-torch==0.2.1
```


## To download and unzip TinyImageNet datasets

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

## To prepare raw CIFAR100, TinyImageNet


Export CIFAR100 as a dataset dir in './data/cifar100':
```
python3 export_raw_data.py --dataset 'CIFAR100' --saved_path './data/cifar100'
```

Export TinyImageNet as a dataset dir in './data/tiny':
```
python3 export_raw_data.py --dataset 'TinyImageNet' --data_path './tiny-imagenet-200' --saved_path './data/tiny'
```

## split data into two halves
```
python3 split_data.py
python3 split_data_tiny.py
```


## train target model
```
python3 train_target_model.py
python3 train_target_model_tiny.py
```