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
from config.student_continer import (student_train,student_val,save_dict_to_json,adjust_learning_rate_cifar,
                                     model_incremental)
from setting import (cifar100_teacher_model_name, teacher_model_path_dict)

split_symbol = '~' if os.name == 'nt' else ':'
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

    parser.add_argument('--model_s', type=str, default='resnet20',
                        choices=[ 'resnet20'])
    parser.add_argument('--model_t', type=str, default='resnet20',
                        choices=['resnet110'])

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    parser.add_argument('--data_ratio', type=str, default='1:1:1',
                        help='Scale the data set for model training,The number of terms in the scale represents the number of teachers')

    parser.add_argument("--teacher_num", type=int, default=3, help='use multiple teacher')
    parser.add_argument('--T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--alpha', type=float, default=0.3, help='temperature for KD distillation')
    opt = parser.parse_args()

    if opt.dataset == 'cifar100':
        opt.teacher_model_name = cifar100_teacher_model_name
    opt.teacher_name_list = [name.split("-")[1]
                             for name in opt.teacher_model_name[:opt.teacher_num]]
    opt.teacher_name_str = "_".join(list(set(opt.teacher_name_list)))

    model_name_template = split_symbol.join(['S', '{}_{}_{}_r', '{}_a', '{}_b', '{}_{}'])
    opt.model_name = model_name_template.format(opt.model_s, opt.dataset, opt.distill,
                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    if opt.teacher_num > 1:
        opt.model_name = opt.model_name + '_' + str(
            opt.teacher_num) + '_' + opt.teacher_name_str + "_" + opt.ensemble_method

    return opt

def load_teacher(model_path, n_cls, model_t, opt=None):
    print('==> loading teacher model')
    model = model_dict[model_t](num_classes=n_cls)
    # TODO: reduce size of the teacher saved in train_teacher.py
    map_location = None if opt.gpu is None else {'cuda:0': 'cuda:%d' % (opt.gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    print('==> done')
    return model

def load_teacher_list(opt, teacher_num, classes_map):
    print('==> loading teacher model list')
    # teacher_model_list = [load_teacher(teacher_model_path_dict[model_name], n_cls, model_t, opt)
    #                       for (model_name, model_t) in zip(opt.teacher_model_name, opt.teacher_name_list)]
    #
    teacher_model_list = []
    for id in range(teacher_num):
        teacher_model_list.append(load_teacher(teacher_model_path_dict[opt.teacher_model_name[id]], classes_map[id+1],
                                               opt.teacher_name_list[id],opt))
    print('==> done')
    return teacher_model_list
best_acc = 0
def main_worker(gpu ,opt):
    global best_acc
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

    save_path = './save/ED/students/'

    opt.model_name = '{}_{}_lr_{}_decay_{}_student_{}'.format(opt.model, opt.dataset,
                                                                                  opt.learning_rate,
                                                                   opt.weight_decay, opt.model_s)

    opt.save_folder = os.path.join(save_path, opt.model)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    model_s = model_dict[opt.model](number_classes=data_numbers[1])
    incremental_learning_blocks2 = model_dict[opt.model_s+'_block2']
    teaching_reference_refer2 = model_dict[opt.model_s+'_reference2'](num_classes = data_numbers[:3])
    incremental_learning_blocks3 = model_dict[opt.model_s + '_block3']
    teaching_reference_refer3 = model_dict[opt.model_s + '_reference3'](num_classes=data_numbers[:4])

    model_t_list = load_teacher_list(opt, opt.teacher_num, data_numbers, )

    optimizer = optim.SGD(model_s.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()
    milestones = list(map(int, opt.lr_decay_epochs.split(',')))
    if len(milestones) < 2 :
        print('number of milestones mast > 2')
        return ModuleNotFoundError()

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        if torch.cuda.device_count() > 1:
            model_s = nn.DataParallel(model_s).cuda()
            teaching_reference_refer2 = nn.DataParallel(teaching_reference_refer2).cuda()
            incremental_learning_blocks3 = nn.DataParallel(incremental_learning_blocks3).cuda()
            teaching_reference_refer3 = nn.DataParallel(teaching_reference_refer3).cuda()
        else:
            model_s = model_s.cuda()
            incremental_learning_blocks2 = incremental_learning_blocks2.cuda()
            teaching_reference_refer2 = teaching_reference_refer2.cuda()

            incremental_learning_blocks3 = incremental_learning_blocks3.cuda()
            teaching_reference_refer3 = teaching_reference_refer3.cuda()
    else:
        print(ModuleNotFoundError)
    block2 = (incremental_learning_blocks2 , teaching_reference_refer2,'block2')
    block3 = (incremental_learning_blocks3 , teaching_reference_refer3,'block3')


    if opt.dataset == 'cifar100':
        primary_train,primary_val = get_cifar100_dataloaders(0 ,num_class=data_numbers,
                                                            batch_size=opt.batch_size, num_workers=opt.num_workers)
        middle_train,middle_val = get_cifar100_dataloaders(1 ,num_class=data_numbers,
                                                            batch_size=opt.batch_size, num_workers=opt.num_workers)
        university_train,university_val = get_cifar100_dataloaders(2 ,num_class=data_numbers,
                                                            batch_size=opt.batch_size, num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)

    for epoch in range(1, opt.epochs + 1):
        if epoch < milestones[0]:

            adjust_learning_rate_cifar(optimizer,epoch,opt)

            train_correct1 , train_loss1, train_total1 = student_train(epoch , model_s, model_t_list[0],primary_train,optimizer,criterion,opt)
            print(f'Train, Loss: {train_correct1 / train_total1:.3f}, Top-1: {train_loss1 / (len(primary_val)):.3f}')

            logger.log_value('train_top1', train_correct1 / train_total1, epoch)
            logger.log_value('train_loss',train_loss1  / (len(primary_val)), epoch)

            test_correct1, test_total1, test_train_loss1 =  student_val(epoch , primary_val, model_s ,criterion,opt)
            print(f'Val, Loss: {test_correct1/test_total1:.3f}, Top-1: {test_train_loss1/(len(primary_val)):.3f}')

        if milestones[0] <= epoch < milestones[1]:
            if epoch == milestones[0]:
                optimizer,model_s = model_incremental(model_s,optimizer,block2)
            adjust_learning_rate_cifar(optimizer, epoch, opt)

            train_correct1, train_loss1, train_total1 = student_train(epoch, model_s, model_t_list[0],
                                                                      primary_train, optimizer, criterion, opt,
                                                                      ratio=(data_numbers[0],data_numbers[1]))
            train_correct2, train_loss2, train_total2 = student_train(epoch, model_s, model_t_list[1],
                                                                      middle_train, optimizer, criterion, opt,
                                                                      ratio=(data_numbers[1],data_numbers[2]))

            print(f'Train, Loss: {(train_correct1+train_correct2) / (train_total1+train_correct2):.3f}, '
                  f'Top-1: {(train_loss1+train_loss2) / (len(primary_val)+len(middle_val)):.3f}')

            logger.log_value('train_top1',(train_correct1+train_correct2) / (train_total1+train_correct2),epoch)
            logger.log_value('train_loss',(train_loss1+train_loss2) / (len(primary_val)+len(middle_val)),epoch)

            test_correct1, test_total1, test_train_loss1 = student_val(epoch, primary_val, model_s, criterion, opt)
            test_correct2, test_total2, test_train_loss2 = student_val(epoch, middle_val, model_s, criterion, opt)
            print(f'Val, Loss: {(test_correct1+test_correct2) / (test_total1+test_total2):.3f},'
                  f' Top-1: {(test_train_loss1+test_train_loss2) / (len(primary_val)+len(middle_val)):.3f}')

            logger.log_value('test_top1', (test_correct1+test_correct2) / (test_total1+test_total2), epoch)
            logger.log_value('test_loss', (test_train_loss1+test_train_loss2) / (len(primary_val)+len(middle_val)), epoch)

        if epoch >= milestones[1]:
            if epoch == milestones[1]:
                optimizer, model_s = model_incremental(model_s, optimizer, block3)
            adjust_learning_rate_cifar(optimizer, epoch, opt)

            train_correct1, train_loss1, train_total1 = student_train(epoch, model_s, model_t_list[0],
                                                                      primary_train, optimizer, criterion, opt,
                                                                      ratio=(data_numbers[0], data_numbers[1]))
            train_correct2, train_loss2, train_total2 = student_train(epoch, model_s, model_t_list[1],
                                                                      middle_train, optimizer, criterion, opt,
                                                                      ratio=(data_numbers[1], data_numbers[2]))
            train_correct3, train_loss3, train_total3 = student_train(epoch, model_s, model_t_list[2],
                                                                      university_train, optimizer, criterion, opt,
                                                                      ratio=(data_numbers[2], data_numbers[3]))
            print(f'Train, Top-1: {(train_correct1 + train_correct2 + train_correct3) /(train_total1 + train_total2 + train_total3):.3f}, '
                  f'Loss: {(train_loss1+train_loss2+train_loss3) / (len(primary_val)+len(middle_val)+len(university_train)):.3f}')

            logger.log_value('train_top1', (train_correct1 + train_correct2 + train_correct3) /
                             (train_total1 + train_total2 + train_total3), epoch)
            logger.log_value('train_loss', (train_loss1+train_loss2+train_loss3) /
                             (len(primary_val)+len(middle_val)+len(university_train)), epoch)

            test_correct1, test_total1, test_train_loss1 = student_val(epoch, primary_val, model_s, criterion, opt)
            test_correct2, test_total2, test_train_loss2 = student_val(epoch, middle_val, model_s, criterion, opt)
            test_correct3, test_total3, test_train_loss3 = student_val(epoch, university_val, model_s, criterion, opt)

            print(f'Val, Top-1: {(test_correct1+test_correct2+test_correct3) / (test_total1+test_total2+test_total3):.3f},'
                  f' Loss: {(test_train_loss1+test_train_loss2+test_train_loss3) /(len(primary_val)+len(middle_val)+len(university_val)):.3f}')

            logger.log_value('test_top1', (test_correct1+test_correct2+test_correct3) /
                             (test_total1+test_total2+test_total3), epoch)
            logger.log_value('test_loss', (test_train_loss1+test_train_loss2+test_train_loss3) /
                             (len(primary_val)+len(middle_val)+len(university_val)),epoch)

            test_acc = (test_correct1+test_correct2+test_correct3) /(test_total1+test_total2+test_total3)
            test_loss = (test_train_loss1+test_train_loss2+test_train_loss3) /(len(primary_val)+len(middle_val)+len(university_val))

            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                }

                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
                test_merics = {'test_loss': test_loss,
                               'test_acc': test_acc,
                               'epoch': epoch}

                save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
                print('saving the best model!')
                torch.save(state, save_file)
# if __name__ == '__main__':
#     main()
