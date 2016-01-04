require 'image'
require 'itorch'
-- os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
-- os.execute('unzip cifar10torchsmall.zip')
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print(trainset)
print(#trainset)

itorch.image(trainset.data[100]) -- display the 100-th image in dataset
print(classes[trainset.label[100]])
