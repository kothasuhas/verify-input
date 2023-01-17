import os, shutil, json, datetime, time

import core.parser as parser
import core.utils as utils
import core.logger as logger
import core.data as data
import core.trainer as trainer

import torch
import pandas as pd

args = parser.args

utils.seed(1)
DATA_DIR = os.path.join(args.data_dir)
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

train_dataset, test_dataset, train_dataloader, test_dataloader = data.load_data(
    DATA_DIR, args.batch_size, use_augmentation=False
)

utils.seed(1)
trainer = trainer.Trainer(args)
logger.log(trainer.model)

reconstruction_function = torch.nn.MSELoss(size_average=False)
def loss_function(x, y, model):
    recon_x, mu, logvar = model(x)
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return (BCE + KLD) / args.batch_size, recon_x

def accuracy(true, preds):
    return None

trainer.criterion = loss_function
trainer.accuracy = accuracy

# Start of training

if args.saved_model:
    trainer.load_model(args.saved_model)
else:
    logger.log('\n\n')
    metrics = pd.DataFrame()

    trainer.init_optimizer(args.num_epochs)    

    old_score = 100.0
    for epoch in range(1, args.num_epochs +1):
        start = time.time()
        logger.log('======= Epoch {} ======='.format(epoch))
        
        last_lr = trainer.scheduler.get_last_lr()[0]
        # last_lr = 0.00001
        
        res = trainer.train(train_dataloader, epoch=epoch)
        
        logger.log('Loss: {:.4f}.\tLR: {:.4f}'.format(res['loss'], last_lr))

        epoch_metrics = {'train_'+k: v for k, v in res.items()}
        epoch_metrics.update({'epoch': epoch, 'lr': last_lr})
        
        if res['loss'] <= old_score:
            old_score = res['loss']
            trainer.save_model(SAVE_BEST_WEIGHTS)
        trainer.save_model(SAVE_LAST_WEIGHTS)
        
        logger.log('Time taken: {}'.format(utils.format_time(time.time()-start)))
        metrics = metrics.append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True)
        metrics.to_csv(os.path.join(LOG_DIR, 'stats.csv'), index=False)

    logger.log('\nTraining completed.')

logger.log('Script Completed.')
