import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import RobertaTokenizer
from tqdm import tqdm
import argparse
from dataset import Dataset, collate_fn
from model import Clip
from utils import show_config, OTFDistributedSampler
from scheduler import cosine_lr


def train_one_epoch(local_rank, epoch, num_batches_per_epoch, model, train_loader, optimizer, scheduler, scaler):
    train_loader = tqdm(train_loader)
    model.train()
    sampler.set_epoch(epoch)  # shuffle data in each gpu
    train_loss, contrastive_loss, mlm_loss, mlm_ent_loss = 0.0, 0.0, 0.0, 0.0

    for i, (batch_text_x, batch_text_y,  batch_graph_x, batch_graph_y) in enumerate(train_loader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        optimizer.zero_grad()

        batch_text_x = {k: v.cuda(local_rank) for k, v in batch_text_x.items()}
        batch_graph_x = {k: v.cuda(local_rank) for k, v in batch_graph_x.items()}

        if scaler is not None:
            with autocast():
                loss, clip_loss, masked_lm_loss, mlm_loss_ent, _ = model(text=batch_text_x, graph=batch_graph_x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)     
            scaler.update()   
        else:
            loss, clip_loss, masked_lm_loss, mlm_loss_ent, _ = model(text=batch_text_x, graph=batch_graph_x)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        contrastive_loss += clip_loss.item()
        mlm_loss += masked_lm_loss.item()
        mlm_ent_loss += mlm_loss_ent.item()

        postfix = {
            'train_loss': '%.5f' % (train_loss / (i + 1)),
            'contrastive_loss': '%.5f' % (contrastive_loss / (i + 1)),
            'mlm_loss': '%.5f' % (mlm_loss / (i + 1)),
            'ent_loss': '%.5f' % (mlm_ent_loss / (i + 1)),
            'lr':'%.6f' % (optimizer.param_groups[0]['lr'])
        } 
        train_loader.set_postfix(log=postfix)
        
        if i % args.save_interval == 0 and i > 0:
            torch.save(model.module.state_dict(), args.save_path + str(i) + '_' + str(epoch))

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local rank for distributed training")
    parser.add_argument('--gpu_num', type=int, default=4, help='Number of GPUs used in training')
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('-epochs', type=int, default=8, help='Number of epochs for training')
    parser.add_argument('-learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('-warmup_ratio', type=float, default=0.1, help='Warmup ratio for training')
    parser.add_argument('-weight_decay', type=float, default=0.01, help='Weight decay for training')
    parser.add_argument('-save_interval', type=int, default=25000, help='Save interval for training')
    parser.add_argument('-save_path', type=str, default='./save_model/model_', help='Save path for training')
    parser.add_argument('-text_dir', type=str, default='./data/output_text', help='Text data directory')
    parser.add_argument('-graph_dir', type=str, default='./data/output_graph', help='Graph data directory')
    

    args = parser.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    print("Current CUDA: ",local_rank)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if local_rank == 0 and torch.cuda.device_count() > 1:
        print("Found", torch.cuda.device_count(), "GPUs!")

    n_gpus = args.gpu_num
    torch.distributed.init_process_group("nccl", world_size=n_gpus, rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)
    n_proc = torch.distributed.get_world_size()

    text_list = []
    for path, _, filenames in os.walk(args.text_dir):
        text_list.extend([os.path.join(path, filename) for filename in filenames])
    text_list.sort()

    graph_list = []
    for path, _, filenames in os.walk(args.graph_dir):
        graph_list.extend([os.path.join(path, filename) for filename in filenames])
    graph_list.sort()

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    mask_id = tokenizer.mask_token_id
    word_vocab_size = tokenizer.vocab_size

    dataset = Dataset(text_list, graph_list, args.local_rank, n_proc, mask_id, word_vocab_size)
    sampler = OTFDistributedSampler(text_list, n_proc, args.local_rank)   
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=sampler)


    model = nn.parallel.DistributedDataParallel(Clip(device).cuda(args.local_rank),
                                                device_ids=[args.local_rank])
    
    NO_DECAY = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.bias', 'layer_norm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in NO_DECAY)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in NO_DECAY)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)

    length = 5000 * len(text_list) # data volume
    num_batches_per_epoch = (length / args.batch_size) / n_gpus
    warmup = num_batches_per_epoch * args.epochs * args.warmup_ratio
    scheduler = cosine_lr(optimizer, args.learning_rate, warmup, num_batches_per_epoch * args.epochs)
    scaler = GradScaler()

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        print('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', optimizer.param_groups[0]['lr']))
        train_one_epoch(args.local_rank, epoch, num_batches_per_epoch, model, loader, optimizer, scheduler, scaler)
        torch.save(model.module.state_dict(), args.save_path + str(epoch))
