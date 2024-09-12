import torch
import json
from config.cifar100 import get_cifar100_dataloaders
import torch.nn.functional as F
import copy
import numpy as np

from ed_cifar100_resnet20_train_resnet110 import linear2


def assignment(model, lower_model):
    student_model = model
    lower_model = lower_model
    student_model_weights = student_model.state_dict()
    lower_model_weights = lower_model.state_dict()
    for name, param in lower_model_weights.items():
        if name in student_model_weights and param.size() == student_model_weights[name].size():
            student_model_weights[name] = param
    student_model.load_state_dict(student_model_weights)
    return student_model

def increment(model, linear, layer, id='block2'):
    student_model = model
    if id == 'block2':
        student_model.layer2 = layer
        student_model.fc = linear
    if id == 'block3':
        student_model.layer3 = layer
        student_model.fc = linear
    return student_model

def student_train(epoch , model_s, model_t,train_loader,optimizer,criterion,opt,ratio=(0,0)):

    print(f'\nEpoch: {epoch}')
    model_s.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.float()

        if opt.gpu is not None:
            inputs = inputs.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            targets = targets.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        with torch.no_grad():
            teacher_preds = model_t(inputs)

        outputs = model_s(inputs)
        student_loss = criterion(outputs, targets)

        if ratio == (0, 0):
            soft_student = F.softmax(outputs / opt.T, dim=1)
        else:
            m, n = ratio
            soft_student = F.softmax(outputs[:, m:n] / opt.T, dim=1)

        soft_teacher = F.softmax(teacher_preds / opt.T, dim=1)
        loss = soft_teacher.mul(-1 * torch.log(soft_student))
        loss = loss.sum(1)
        distillation_loss = loss.mean() * opt.T * opt.T
        loss = opt.alpha * student_loss + (1 - opt.alpha) * distillation_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        top1 = correct / total
        top1 = round(top1, 3)
        loss = train_loss / (batch_idx + 1)
        loss = round(loss, 3)
        if batch_idx % opt.print_freq == 0:
            print(f'Train:Batch {batch_idx}, Loss: {loss}, Top-1: {top1}')

    return correct , train_loss, total

def student_val(epoch , val_loader, model ,criterion,opt):
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.float()

        if opt.gpu is not None:
            inputs = inputs.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            targets = targets.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return correct , total , train_loss

def model_incremental(model_s, optimizer, block):
    linear,layer, block_id = block
    lower_student_model = copy.deepcopy(model_s)
    student_model = increment(lower_student_model, linear, layer, id= block_id)
    student_model = assignment(student_model, lower_student_model)
    lr = optimizer.param_groups[0]['lr']
    mo = optimizer.param_groups[0].get('momentum', 0)
    wei = optimizer.param_groups[0]['weight_decay']
    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=mo, weight_decay=wei)

    return optimizer,student_model

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def adjust_learning_rate_cifar(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr