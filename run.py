import os, shutil, json, datetime, time

import core.parser as parser
import core.utils as utils
import core.logger as logger
import core.data as data
import core.trainer as trainer

import torch
import pandas as pd
import random
import numpy as np

args = parser.args

utils.seed(1)
DATA_DIR = os.path.join(args.data_dir, 'cifar10s')
LOG_DIR = os.path.join(args.log_dir, datetime.datetime.now().strftime("%m-%d-%H:%M:%S--") + args.log_desc)
SAVE_BEST_WEIGHTS  = os.path.join(LOG_DIR, 'weights-best.pt')
SAVE_LAST_WEIGHTS  = os.path.join(LOG_DIR, 'weights-last.pt')

if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)
logger = logger.Logger(os.path.join(LOG_DIR, 'log-train.log'))

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))

torch.backends.cudnn.benchmark = True

using_cross_entropy = True

if using_cross_entropy:

    m1 = torch.distributions.MultivariateNormal(torch.Tensor([-1,0]), torch.Tensor([[0.2, 0], [0, 0.2]]))
    m2 = torch.distributions.MultivariateNormal(torch.Tensor([1,0]), torch.Tensor([[0.2, 0], [0, 0.2]]))

    num_indist = 200
    m1_samples = [m1.sample().reshape(1, 2) for _ in range(num_indist)]
    m2_samples = [m2.sample().reshape(1, 2) for _ in range(num_indist)]
    id_samples = m1_samples + m2_samples

    # m3_samples = []
    m3_samples = [(4 * torch.rand(1, 2) - 2) for _ in range(300)]
    far = lambda sample: (lambda sample2 : torch.norm(sample - sample2, p=np.inf) > 0.2)
    far_all = lambda samples: (lambda sample: all(list(map(far(sample), samples))))
    m3_samples = list(filter(lambda sample: far_all(m1_samples)(sample) and far_all(m2_samples)(sample), m3_samples))
    print(len(m3_samples))

    x = m1_samples + m2_samples + m3_samples
    y_id = [torch.LongTensor([0 if i < num_indist else 1]).unsqueeze(dim=0).reshape(1) for i in range(2 * num_indist)]
    y_ood = [torch.LongTensor([2]).unsqueeze(dim=0).reshape(1) for i in range(len(m3_samples))]
    y = y_id + y_ood

    assert len(x) == len(y)
    train_dataloader = list(zip(x, y))
    random.shuffle(train_dataloader)

    print(train_dataloader)
    print(y)

    m1_samples = [m1.sample().reshape(1, 2) for _ in range(40)]
    m2_samples = [m2.sample().reshape(1, 2) for _ in range(40)]
    x = list(sum(zip(m1_samples, m2_samples), ()))
    y = [torch.LongTensor([i % 2]).unsqueeze(dim=0).reshape(1) for i in range(200)]
    test_dataloader = list(zip(x, y))

else:

    m1 = torch.distributions.MultivariateNormal(torch.Tensor([-1,0]), torch.Tensor([[0.2, 0], [0, 1]]))
    m2 = torch.distributions.MultivariateNormal(torch.Tensor([1,0]), torch.Tensor([[0.2, 0], [0, 1]]))

    num_indist = 40
    m1_samples = [m1.sample().reshape(1, 2) for _ in range(num_indist)]
    m2_samples = [m2.sample().reshape(1, 2) for _ in range(num_indist)]
    id_samples = m1_samples + m2_samples

    # m3_samples = []
    m3_samples = [(4 * torch.rand(1, 2) - 2) for _ in range(50)]
    far = lambda sample: (lambda sample2 : torch.norm(sample - sample2, p=np.inf) > 0.2)
    far_all = lambda samples: (lambda sample: all(list(map(far(sample), samples))))
    m3_samples = list(filter(lambda sample: far_all(m1_samples)(sample) and far_all(m2_samples)(sample), m3_samples))
    print(len(m3_samples))

    x = m1_samples + m2_samples + m3_samples
    y_id = [torch.LongTensor([1, 0] if i > num_indist else [0, 1]).reshape(1, 2) for i in range(2 * num_indist)]
    y_ood = [torch.LongTensor([0.5, 0.5]).reshape(1, 2) for i in range(len(m3_samples))]
    y = y_id + y_ood

    assert len(x) == len(y)
    train_dataloader = list(zip(x, y))
    random.shuffle(train_dataloader)

    print(train_dataloader)
    print(y)

    m1_samples = [m1.sample().reshape(1, 2) for _ in range(40)]
    m2_samples = [m2.sample().reshape(1, 2) for _ in range(40)]
    x = list(sum(zip(m1_samples, m2_samples), ()))
    y = [torch.LongTensor([i % 2]).unsqueeze(dim=0).reshape(1) for i in range(200)]
    test_dataloader = list(zip(x, y))

utils.seed(1)
trainer = trainer.Trainer(args)
logger.log(trainer.model)

# Start of training

if args.saved_model:
    trainer.load_model(args.saved_model)

    logger.log('\n\n')
    logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader)*100))
else:
    logger.log('\n\n')
    metrics = pd.DataFrame()
    logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader)*100))

    trainer.init_optimizer(args.num_epochs)    

    old_score = 0.0
    for epoch in range(1, args.num_epochs +1):
        start = time.time()
        logger.log('======= Epoch {} ======='.format(epoch))
        
        last_lr = trainer.scheduler.get_last_lr()[0]
        
        res = trainer.train(train_dataloader, epoch=epoch)
        test_acc = trainer.eval(test_dataloader)
        
        logger.log('Loss: {:.4f}.\tLR: {:.4f}'.format(res['loss'], last_lr))
        if 'clean_acc' in res:
            logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['clean_acc']*100, test_acc*100))
        else:
            logger.log('Standard Accuracy-\tTest: {:.2f}%.'.format(test_acc*100))

        epoch_metrics = {'train_'+k: v for k, v in res.items()}
        epoch_metrics.update({'epoch': epoch, 'lr': last_lr, 'test_clean_acc': test_acc})
        
        if test_acc >= old_score:
            old_score = test_acc
            trainer.save_model(SAVE_BEST_WEIGHTS)
        trainer.save_model(SAVE_LAST_WEIGHTS)
        
        logger.log('Time taken: {}'.format(utils.format_time(time.time()-start)))
        metrics = metrics.append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True)
        metrics.to_csv(os.path.join(LOG_DIR, 'stats.csv'), index=False)

    # Record metrics

    train_acc = res['clean_acc'] if 'clean_acc' in res else trainer.eval(train_dataloader)
    logger.log('\nTraining completed.')
    logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(train_acc*100, old_score*100))

logger.log('Script Completed.')
