# echo "Running Inception Cifar10"
# python run_bottleneck.py --network inception --batch_size 32 --dataset cifar10
# echo "Running Inception Traffic"
# python run_bottleneck.py --network inception --batch_size 32 --dataset traffic
# echo "Running ResNet Cifar10"
# python run_bottleneck.py --network resnet --batch_size 32 --dataset cifar10
# echo "Running ResNet Traffic"
# python run_bottleneck.py --network resnet --batch_size 32 --dataset traffic
# echo "Running VGG Cifar10"
# python run_bottleneck.py --network vgg --batch_size 16 --dataset cifar10
# echo "Running VGG Traffic"
# python run_bottleneck.py --network vgg --batch_size 16 --dataset traffic
python shrink.py --network vgg --dataset traffic --size 100
python shrink.py --network vgg --dataset cifar10 --size 100
python shrink.py --network resnet --dataset traffic --size 100
python shrink.py --network resnet --dataset cifar10 --size 100
python shrink.py --network inception --dataset traffic --size 100
python shrink.py --network inception --dataset cifar10 --size 100