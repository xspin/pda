import torch
import torchvision
from config import config
from net import Net
import data
import numpy as np
from itertools import cycle
from timer import Clock

def accuracy(y_true, y_pred):
    y_pred = torch.argmax(y_pred, axis=-1)
    return torch.sum(torch.eq(y_true, y_pred), dtype=torch.float)/len(y_true)

dataset_amazon_dir = 'datasets/office31/amazon/images'
dataset_dslr_dir = 'datasets/office31/dslr/images'
dataset_webcam_dir = 'datasets/office31/webcam/images'
dataset_dir = {'amazon': dataset_amazon_dir, 'dslr': dataset_dslr_dir, 'webcam': dataset_webcam_dir}

#! Modify here to change the source and target domain
src_domain_name = 'amazon'
tgt_domain_name = 'dslr'

config.epochs = 3
config.batch_size = 16
config.base_model = 'resnet'
config.image_size = (256, 256)
config.classes = 31

src_domain_dir = dataset_dir[src_domain_name]
tgt_domain_dir = dataset_dir[tgt_domain_name]

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(config.image_size),
    # RandomCrop(224),
    # RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])

print('Loading dataset ...')
src_dataset = data.MyDataset(path=src_domain_dir, labels='all', transform=transform)
label2index = src_dataset.label2index
tgt_dataset = data.MyDataset(path=tgt_domain_dir, 
                        labels=data.shared_classes,
                        transform=transform, 
                        label2index=label2index)

config.classes = len(src_dataset.label2index)
print('src size:', len(src_dataset))
print('tgt size:', len(tgt_dataset))

src_dataloader = torch.utils.data.DataLoader(src_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)
tgt_dataloader = torch.utils.data.DataLoader(tgt_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)

print('train batchs:', len(src_dataloader))
print('test batchs:', len(tgt_dataloader))
src_dis_labels = torch.zeros(config.batch_size, dtype=torch.long)
tgt_dis_labels = torch.ones(config.batch_size, dtype=torch.long)

print('Initializing network ...')
net = Net(config)
criterion_label = torch.nn.CrossEntropyLoss() 
criterion_domain = torch.nn.CrossEntropyLoss() 
# optimizer_F = torch.optim.SGD(net.feature_extractor.parameters(), lr=0.001, momentum=0.9)
# optimizer_D = torch.optim.SGD(net.adversarial_classifier.parameters(), lr=0.001, momentum=0.9)
optimizer_F = torch.optim.Adam(net.feature_extractor.parameters())
optimizer_D = torch.optim.Adam(net.adversarial_classifier.parameters())

print('Starting training ...')
clock_epoch = Clock(config.epochs)
for epoch in range(config.epochs):
    step = 0
    clock_batch = Clock(len(src_dataloader))
    for src_data, tgt_data in zip(src_dataloader, cycle(tgt_dataloader)):
        print('  Epoch {}/{}  Batch {}/{}'.format(epoch+1, config.epochs, step+1, len(src_dataloader)))
        src_inputs, src_labels = src_data
        tgt_inputs, tgt_labels = tgt_data
        src_inputs = torch.autograd.Variable(src_inputs)
        tgt_inputs = torch.autograd.Variable(tgt_inputs)
        optimizer_F.zero_grad()
        src_y_label, src_y_domain = net(src_inputs)
        tgt_y_label, tgt_y_domain = net(tgt_inputs)
        src_loss_label = criterion_label(src_y_label, src_labels)
        # tgt_loss_label = criterion_label(tgt_y_label, tgt_labels)
        src_loss_domain = criterion_domain(src_y_domain, src_dis_labels)
        tgt_loss_domain = criterion_domain(tgt_y_domain, tgt_dis_labels)
        loss = src_loss_label - src_loss_domain - tgt_loss_domain
        loss.backward(retain_graph=True)        
        optimizer_F.step()

        optimizer_D.zero_grad()
        loss = src_loss_label + src_loss_domain + tgt_loss_domain
        loss.backward()        
        optimizer_D.step()
        
        # print statistics
        src_loss_lb = src_loss_label.item()
        src_loss_dm = src_loss_domain.item()
        tgt_loss_dm = tgt_loss_domain.item()
        src_acc_lb = accuracy(src_labels, src_y_label)
        tgt_acc_lb = accuracy(tgt_labels, tgt_y_label)
        src_acc_dm = accuracy(src_dis_labels, src_y_domain)
        tgt_acc_dm = accuracy(tgt_dis_labels, tgt_y_domain)

        print('\tacc_label {} {}  acc_domain {} {}'.format(src_acc_lb, tgt_acc_lb, src_acc_dm, tgt_acc_dm))
        print('\tloss_label {:.6f}  loss_domain {:.6f} {:.6f}'.format(src_loss_label, tgt_loss_dm, src_loss_dm, tgt_loss_dm))
        tm = clock_batch.toc(step)
        print('\tElapsed {}  ETA {}'.format(*tm))
        step += 1
    tm = clock_epoch.toc(epoch)
    print('  Elapsed {}  ETA {}'.format(*tm))
print('Finished Training')