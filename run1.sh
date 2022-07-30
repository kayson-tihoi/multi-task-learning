# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --dataset mini-imagenet --model modified_resnet18 

# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 1 --algorithm mtl --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 1 --algorithm mtl --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 1 --algorithm mtl --dataset mini-imagenet --model modified_resnet18 

# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --dataset mini-imagenet --model modified_resnet18 --epoch_length 50 --val_epoch_length 10 --test_epoch_length 50
# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --dataset mini-imagenet --model modified_resnet18 --epoch_length 50 --val_epoch_length 10 --test_epoch_length 50
# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --dataset mini-imagenet --model modified_resnet18 --epoch_length 50 --val_epoch_length 10 --test_epoch_length 50

set -ex

python train.py --gpu 1 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots  1 --algorithm mtl --dataset mini-imagenet --epoch_length 50 --val_epoch_length 10 --model modified_resnet34 
python train.py --gpu 1 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots  1 --algorithm mtl --dataset mini-imagenet --epoch_length 50 --val_epoch_length 10 --model modified_resnet34 
python train.py --gpu 1 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots  1 --algorithm mtl --dataset mini-imagenet --epoch_length 50 --val_epoch_length 10 --model modified_resnet34 

python train.py --gpu 1 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots  5 --algorithm mtl --dataset mini-imagenet --epoch_length 50 --val_epoch_length 10 --model modified_resnet34 
python train.py --gpu 1 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots  5 --algorithm mtl --dataset mini-imagenet --epoch_length 50 --val_epoch_length 10 --model modified_resnet34 
python train.py --gpu 1 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots  5 --algorithm mtl --dataset mini-imagenet --epoch_length 50 --val_epoch_length 10 --model modified_resnet34 

python train.py --gpu 1 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 10 --algorithm mtl --dataset mini-imagenet --epoch_length 50 --val_epoch_length 10 --model modified_resnet34 
python train.py --gpu 1 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 10 --algorithm mtl --dataset mini-imagenet --epoch_length 50 --val_epoch_length 10 --model modified_resnet34 
python train.py --gpu 1 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 10 --algorithm mtl --dataset mini-imagenet --epoch_length 50 --val_epoch_length 10 --model modified_resnet34 

