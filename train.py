import torch
import torchvision
from config import config
from net import Net
import data

dataset_amazon_dir = 'datasets/office31/amazon/images'
dataset_dslr_dir = 'datasets/office31/dslr/images'
dataset_webcam_dir = 'datasets/office31/webcam/images'

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(config.image_size),
    # RandomCrop(224),
    # RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])

print('Loading dataset ...')
dataset = data.MyDataset(path=dataset_amazon_dir, labels='all', transform=transform)
config.classes = len(dataset.label2index)
n = len(dataset)
n_train = int(n*0.8)
n_test = n - n_train
train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test])
print('train:', len(train_ds))
print('test:', len(test_ds))

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)
print('train batchs:', len(train_dl))
print('test batchs:', len(test_dl))

print('Initializing network ...')
net = Net(config)
criterion = torch.nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('Starting training...')
for epoch in range(config.epochs):
    running_loss = 0.0
    for i,data in enumerate(train_dl, 0):
        inputs, labels = data
        inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()        
        optimizer.step()
        
        # print statistics
        running_loss = loss.item()
#         if (i+1) % 1 == 0: 
        print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 100))
#             running_loss = 0.0
print('Finished Training')