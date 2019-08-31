import torch
import torchvision
from config import config
from net import Net
import data
import numpy as np
from itertools import cycle
from timer import Clock
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='log.txt', level=logging.INFO, format=LOG_FORMAT)

def accuracy(y_true, y_pred, return_count=False):
    y_pred = torch.argmax(y_pred, axis=-1)
    if return_count:
        return torch.sum(torch.eq(y_true, y_pred), dtype=torch.float)
    return torch.sum(torch.eq(y_true, y_pred), dtype=torch.float)/len(y_true)

if __name__ == "__main__":
    #* Modify here to change the source and target domain, and also other configs
    config.src_domain_name = 'amazon'
    config.tgt_domain_name = 'dslr'
    config.epochs = 3
    config.batch_size = 4
    config.base_model = 'resnet'
    config.image_size = (256, 256)
    config.is_cuda = torch.cuda.is_available()

    logging.info('==== config start ====')
    for param in config.__dict__.items():
        logging.info('{}: {}'.format(*param))
    logging.info('==== config end ====')

    src_domain_dir = data.dataset_dir[config.src_domain_name]
    tgt_domain_dir = data.dataset_dir[config.tgt_domain_name]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config.image_size),
        # RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config.image_size),
        # CenterCrop(224),
        torchvision.transforms.ToTensor()
    ])

    print('Loading dataset ...')
    src_dataset = data.MyDataset(path=src_domain_dir, labels='all', transform=transform)
    label2index = src_dataset.label2index
    tgt_dataset = data.MyDataset(path=tgt_domain_dir, 
                            labels=data.shared_classes,
                            transform=transform, 
                            label2index=label2index)
    src_test_dataset = data.MyDataset(path=src_domain_dir, 
                            labels='all',
                            transform=test_transform, 
                            label2index=label2index)
    tgt_test_dataset = data.MyDataset(path=tgt_domain_dir, 
                            labels=data.shared_classes,
                            transform=test_transform, 
                            label2index=label2index)

    config.classes = len(src_dataset.label2index)
    print('src size:', len(src_dataset))
    print('tgt size:', len(tgt_dataset))

    src_dataloader = torch.utils.data.DataLoader(src_dataset, 
                            batch_size=config.batch_size, 
                            shuffle=True, 
                            num_workers=2, 
                            drop_last=True,
                            pin_memory=config.is_cuda)
    tgt_dataloader = torch.utils.data.DataLoader(tgt_dataset, 
                            batch_size=config.batch_size, 
                            shuffle=True, 
                            num_workers=2, 
                            drop_last=True,
                            pin_memory=config.is_cuda)
    src_test_dataloader = torch.utils.data.DataLoader(src_test_dataset, 
                            batch_size=config.batch_size, 
                            shuffle=False, 
                            num_workers=2, 
                            drop_last=False,
                            pin_memory=config.is_cuda)
    tgt_test_dataloader = torch.utils.data.DataLoader(tgt_test_dataset, 
                            batch_size=config.batch_size, 
                            shuffle=False, 
                            num_workers=2, 
                            drop_last=False,
                            pin_memory=config.is_cuda)

    print('train batchs:', len(src_dataloader))
    print('test batchs:', len(tgt_dataloader))

    if config.is_cuda:
        src_dis_labels = torch.zeros(config.batch_size, dtype=torch.long).cuda()
        tgt_dis_labels = torch.ones(config.batch_size, dtype=torch.long).cuda()
    else:
        src_dis_labels = torch.zeros(config.batch_size, dtype=torch.long)
        tgt_dis_labels = torch.ones(config.batch_size, dtype=torch.long)

    print('Initializing network ...')
    if config.is_cuda:
        net = Net(config).cuda()
    else:
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
        print(' Epoch {}/{}'.format(epoch+1, config.epochs))
        step = 0
        clock_batch = Clock(len(src_dataloader))
        for src_data, tgt_data in zip(src_dataloader, cycle(tgt_dataloader)):
            print('    Batch {}/{}'.format(step+1, len(src_dataloader)))
            src_inputs, src_labels = src_data
            tgt_inputs, tgt_labels = tgt_data
            # src_inputs = torch.autograd.Variable(src_inputs)
            # tgt_inputs = torch.autograd.Variable(tgt_inputs)
            if config.is_cuda:
                src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()
                tgt_inputs, tgt_labels = tgt_inputs.cuda(), tgt_labels.cuda()
            else:
                src_inputs, src_labels = src_inputs, src_labels
                tgt_inputs, tgt_labels = tgt_inputs, tgt_labels
            src_y_label, src_y_domain = net(src_inputs)
            tgt_y_label, tgt_y_domain = net(tgt_inputs)
            src_loss_label = criterion_label(src_y_label, src_labels)
            # tgt_loss_label = criterion_label(tgt_y_label, tgt_labels)
            src_loss_domain = criterion_domain(src_y_domain, src_dis_labels)
            tgt_loss_domain = criterion_domain(tgt_y_domain, tgt_dis_labels)
            src_loss_domain_inv = criterion_domain(src_y_domain, tgt_dis_labels)
            tgt_loss_domain_inv = criterion_domain(tgt_y_domain, src_dis_labels)
            loss_F = src_loss_label + src_loss_domain_inv + tgt_loss_domain_inv
            loss_D = src_loss_label + src_loss_domain + tgt_loss_domain

            optimizer_F.zero_grad()
            loss_F.backward(retain_graph=True)        
            optimizer_F.step()

            optimizer_D.zero_grad()
            loss_D.backward()        
            optimizer_D.step()
            
            # print statistics
            src_loss_lb = src_loss_label.item()
            src_loss_dm = src_loss_domain.item()
            tgt_loss_dm = tgt_loss_domain.item()
            src_acc_lb = accuracy(src_labels, src_y_label)
            tgt_acc_lb = accuracy(tgt_labels, tgt_y_label)
            src_acc_dm = accuracy(src_dis_labels, src_y_domain)
            tgt_acc_dm = accuracy(tgt_dis_labels, tgt_y_domain)

            print('\tacc_label {} {}  acc_domain {} {}'.format(src_acc_lb, tgt_acc_lb, src_acc_dm, tgt_acc_dm), end='  ')
            print('loss_label {:.6f}  loss_domain {:.6f} {:.6f}'.format(src_loss_label, src_loss_dm, tgt_loss_dm))
            tm = clock_batch.toc(step)
            avgtm = clock_batch.avg()
            print('\tElapsed {}  ETA {}  AVG {}/batch'.format(*tm, avgtm))

        # Testing
        print('    Testing ...')
        with torch.no_grad():
            for k, dl in enumerate([tgt_test_dataloader, src_test_dataloader]):
                test_clock = Clock()
                cnt = 0.0
                domain = 'tgt' if k==0 else 'src'
                print('\t{}'.format(domain), end='  ')
                for i, data in enumerate(dl):
                    inputs, labels = data
                    if config.is_cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    else:
                        inputs, labels = inputs, labels
                    y_label, y_domain = net(inputs)
                    cnt += accuracy(labels, y_label, return_count=True)
                    if (i+1)%(len(dl)//10)==0:
                        progress = (i+1.0)/len(dl)*100
                        print('{:.0f}%'.format(progress), end=' ', flush=True)
                print()
                acc = cnt/len(dl)
                tm = test_clock.toc()
                print('\t{}_acc  {:.4f}  Elapsed {}'.format(domain, acc, tm))
                logging.info('{}_acc {}'.format(domain, acc))

        tm = clock_epoch.toc(epoch)
        print('  Elapsed {}  ETA {}'.format(*tm))
    print('Finished Training')