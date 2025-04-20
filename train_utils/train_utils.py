def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                   lr_scheduler=None, print_freq=10, scaler=None, use_pk_maps=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device)
        target = target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        # ... 其余代码保持不变 ...

def evaluate(model, data_loader, device, num_classes, use_pk_maps=False):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric = utils.DiceCoefficient(num_classes=num_classes-1, ignore_index=255)
    
    with torch.no_grad():
        for image, target in tqdm(data_loader, desc="Validation..."):
            image = image.to(device)
            target = target.to(device)
            
            output = model(image)
            
            # ... 其余代码保持不变 ...