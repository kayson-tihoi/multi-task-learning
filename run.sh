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

# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset cifarfs --model modified_resnet20 
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset cifarfs --model modified_resnet32
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet34 --image_size 224
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 10 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 10 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet34 --image_size 224

# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 5  --algorithm mtl --epoch_length 100 --val_epoch_length 20 --dataset cifarfs --model modified_resnet20 
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 5  --algorithm mtl --epoch_length 100 --val_epoch_length 20 --dataset cifarfs --model modified_resnet32
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 10 --algorithm mtl --epoch_length 100 --val_epoch_length 20 --dataset cifarfs --model modified_resnet20 
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 10 --algorithm mtl --epoch_length 100 --val_epoch_length 20 --dataset cifarfs --model modified_resnet32

python train.py --gpu 0 --root /data/home/tiankai/data --norm_train_features --meta_batch_size 2 --test_shots 5  --algorithm mtl --epoch_length 100 --val_epoch_length 20 --dataset cifarfs --model ours_vgg 



# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm protonet --epoch_length 50 --val_epoch_length 10 --dataset cifarfs --model modified_resnet20 
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm protonet --epoch_length 50 --val_epoch_length 10 --dataset cifarfs --model modified_resnet32
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm protonet --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm protonet --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet34 --image_size 224
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 10 --algorithm protonet --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root /home/v-tiahang/data --norm_train_features --meta_batch_size 2 --test_shots 10 --algorithm protonet --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet34 --image_size 224


# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 5 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 

# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 20 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 20 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 20 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 

# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 30 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 30 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 
# python train.py --gpu 0 --root ~/data --norm_train_features --meta_batch_size 2 --test_shots 30 --algorithm mtl --epoch_length 50 --val_epoch_length 10 --dataset mini-imagenet --model modified_resnet18 
