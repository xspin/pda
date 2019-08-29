import torch
import torchvision
from config import config
from net import Net
import data
import numpy as np
from itertools import cycle

def accuracy(y_true, y_pred):
    y_pred = torch.argmax(y_pred, axis=-1)
    return torch.sum(torch.eq(y_true, y_pred), dtype=torch.float)/len(y_true)

dataset_amazon_dir = 'datasets/office31/amazon/images'
dataset_dslr_dir = 'datasets/office31/dslr/images'
dataset_webcam_dir = 'datasets/office31/webcam/images'

dataset_dir = {'amazon': dataset_amazon_dir, 'dslr': dataset_dslr_dir, 'webcam': dataset_webcam_dir}

src_domain_name = 'amazon'
tgt_domain_name = 'dslr'

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
# n = len(dataset)
# n_train = int(n*0.8)
# n_test = n - n_train
# train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test])
print('src:', len(src_dataset))
print('tgt:', len(tgt_dataset))

src_dataloader = torch.utils.data.DataLoader(src_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
tgt_dataloader = torch.utils.data.DataLoader(tgt_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

print('train batchs:', len(src_dataloader))
print('test batchs:', len(tgt_dataloader))
src_dis_labels = torch.zeros(config.batch_size, dtype=torch.long)
tgt_dis_labels = torch.ones(config.batch_size, dtype=torch.long)
# dis_src_labels = torch.cat([vec_ones, vec_zeros], dim=-1)
# dis_tgt_labels = torch.cat([vec_zeros, vec_ones], dim=-1)

print('Initializing network ...')
net = Net(config)
criterion_label = torch.nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
criterion_domain = torch.nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer_A = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer_F = torch.optim.SGD(net.feature_extractor.parameters(), lr=0.001, momentum=0.9)
optimizer_D = torch.optim.SGD(net.adversarial_classifier.parameters(), lr=0.001, momentum=0.9)

print('Starting training...')
for epoch in range(config.epochs):
    running_loss = 0.0
    step = 0
    for src_data, tgt_data in zip(cycle(src_dataloader), tgt_dataloader):
        step += 1
        print('Epoch {}/{} Batch {}/{}'.format(epoch, config.epochs, step, len(src_dataloader)))
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
#         if (i+1) % 1 == 0: 
        # print('Epoch {} Batch {} acc: {} {} loss: {} {} {}'.format(epoch+1, step, acc_lb, acc_dm, losslb, lossdm))
        print('  acc_label {} {} acc_domain {} {}'.format(src_acc_lb, tgt_acc_lb, src_acc_dm, tgt_acc_dm))
        print('  loss_label {} loss_domain {} {}'.format(src_loss_label, tgt_loss_dm, src_loss_dm, tgt_loss_dm))
#             running_loss = 0.0
print('Finished Training')