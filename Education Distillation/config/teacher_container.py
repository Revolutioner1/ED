import torch
import json
def teacher_train(epoch , train_loader, model , optimizer,criterion,opt,teacher_id,each_classes):
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.float()

        if opt.gpu is not None:
            inputs = inputs.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            targets = targets.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        targets = targets - sum(each_classes[:teacher_id])
        outputs = model(inputs)
        loss = criterion(outputs, targets)

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
    top1 = correct / total
    top1 = round(top1, 3)
    loss = train_loss / (len(train_loader))
    loss = round(loss, 3)
    return top1 , loss

def teacher_val(epoch , val_loader, model ,criterion,opt,teacher_id,each_classes):
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

        targets = targets - sum(each_classes[:teacher_id])
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    top1 = correct / total
    top1 = round(top1, 3)
    loss = train_loss / (len(val_loader))
    loss = round(loss, 3)

    print(f'Val, Loss: {loss}, Top-1: {top1}')

    return top1 , loss

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