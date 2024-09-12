import os
import argparse
import math
from model import model_dict
import torch.optim as optim
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from config.cifar100 import get_cifar100_dataloaders
import tensorboard_logger as tb_logger
from config.teacher_container import teacher_train,teacher_val,save_dict_to_json

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # baisc
    parser.add_argument('--print-freq', type=int, default=200, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--model', type=str, default='resnet110',
                        choices=[ 'resnet110'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100','caltech256','tinyimagenet'], help='dataset')
    parser.add_argument('-t', '--trial', type=str, default='0', help='the experiment id')
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)

    parser.add_argument('--data_ratio', type=str, default='1:1:1', help='Scale the data set for model training,The number of terms in the scale represents the number of teachers')

    opt = parser.parse_args()

    return opt

def main_worker(gpu ,opt):
    # opt = parse_option()
    data_ratio = opt.data_ratio
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    n_cls = {
        'cifar100':100
    }

    number_classes = n_cls[opt.dataset]
    data_ratio = list(map(int, data_ratio))
    teacher_numbers = len(data_ratio)
    data_numbers = []
    for i in range(len(data_ratio)):
        data_numbers.append(math.floor(number_classes/teacher_numbers) * data_ratio[i])
    data_numbers[-1] = data_ratio[-1] + number_classes - sum(data_numbers)
    data_numbers.append(number_classes)
    data_numbers.insert(0,0)

    for i in range(1,teacher_numbers+1):
        best_acc = 0

        save_path = './save/ED/teachers/'

        opt.model_name = '{}_{}_lr_{}_decay_{}_teacher_id_{}_train_classes_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                                opt.weight_decay, i, data_numbers[i])

        opt.save_folder = os.path.join(save_path, opt.model)

        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
        model = model_dict[opt.model](number_classes=data_numbers[i])
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
        criterion = nn.CrossEntropyLoss()

        milestones = list(map(int, opt.lr_decay_epochs.split(',')))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        if torch.cuda.is_available():
            criterion = criterion.cuda()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model).cuda()
            else:
                model = model.cuda()
        cudnn.benchmark = True

        if opt.dataset == 'cifar100':
            train_loader, val_loader = get_cifar100_dataloaders(teacher_id=i,num_class=data_numbers,
                                                                batch_size=opt.batch_size, num_workers=opt.num_workers)
        else:
            raise NotImplementedError(opt.dataset)

        for epoch in range(1, opt.epochs + 1):
            top1,loss = teacher_train(epoch , train_loader, model , optimizer,criterion,opt, i, data_numbers)
            print('Train: * Epoch {}, Acc@1 {:.3f},Loos {:.3f}'.format(epoch, top1, loss))

            logger.log_value('train_acc', top1, epoch)
            logger.log_value('train_loss', loss, epoch)

            val_top1,val_loss = teacher_val(epoch , val_loader, model ,criterion,opt, i, data_numbers)
            logger.log_value('val_acc', val_top1, epoch)
            logger.log_value('val_loss', val_loss, epoch)
            print('Val: * Epoch {}, Acc@1 {:.3f},Loos {:.3f}'.format(epoch, val_top1, val_loss))

            scheduler.step(epoch)

            if val_top1 > best_acc:
                best_acc = val_top1
                state = {
                    'epoch': epoch,
                    'model': model.module.state_dict() if opt.multiprocessing_distributed else model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))

                test_merics = {'test_loss': float('%.3f' % val_loss),
                               'test_acc': float('%.3f' % val_top1),
                               'epoch': epoch}

                save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))

                print('saving the best model!')
                torch.save(state, save_file)
def main():
    opt = parse_option()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node

    main_worker(None if ngpus_per_node > 1 else opt.gpu_id, opt)
if __name__ == '__main__':
    main()