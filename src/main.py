'''
***********************************************************************
CatDive: A Simple yet Effective Method for Maximizing Category Diversity in Sequential Recommendation

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: main.py
- A main class for CatDive.

Version: 1.0
***********************************************************************
'''


import os
import time
import torch
import argparse
from tqdm import tqdm
from model import SASRec
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='books', type=str)
parser.add_argument('--dir', default='1', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--seed', default=3, type=int)
parser.add_argument('--hidden_units', default=128, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=10000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.001, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--test', default=False, type=bool)
parser.add_argument('--topk', default=10, type=int)
parser.add_argument('--alpha', default=0, type=float) # alpha of Multi-Embedding (see Equation (3))
parser.add_argument('--beta', default=-1, type=float) # beta of Category-Weighted High-Confidence Negative Sampling (see Equation (7)), -1 if using random sampling
parser.add_argument('--lamb', default=0, type=float) # lambda of Coverage-Prioritized Reranking (see Equation (10))


args = parser.parse_args()
if not os.path.isdir('runs/'+args.dataset + '/' + args.dir):
    os.makedirs('runs/'+args.dataset + '/' + args.dir)

if args.dataset == 'books':
    args.maxlen = 50
elif args.dataset == 'kindle':
    args.maxlen = 50
elif args.dataset == 'gr-r':
    args.maxlen = 150

if __name__ == '__main__':
    setup_seed(args.seed)
    dataset = data_partition(args.dataset)
    
    [user_train, user_valid, user_test, usernum, itemnum, catenum, category, popularity] = dataset # loads data with category and popularity information
    num_batch = len(user_train) // args.batch_size 
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
        
    # prints data information
    print('\ndataset:', args.dataset)
    print('name:', args.dir)
    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('user num:', usernum)
    print('item num:', itemnum)
    print('category num:', catenum)
    
    sampler = WarpSampler(user_train, usernum, itemnum, catenum, category, popularity, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=4, beta=args.beta)
    
    model = SASRec(usernum, itemnum, catenum, category, args.num_blocks, args.num_heads, args.dropout_rate, args.hidden_units, args.maxlen, args.device, alpha=args.alpha).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass 
    
    model.train() 
    
    epoch_start_idx = 1
    
    
    # train or test
    if args.test:
        f = open(os.path.join('runs/'+args.dataset + '/' + args.dir, 'log.txt'), 'a')
    else:
        f = open(os.path.join('runs/'+args.dataset + '/' + args.dir, 'log.txt'), 'w')
    
    bce_criterion = torch.nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    
    T = 0.0
    t0 = time.time()
    best = 0
    
    
    # training
    p = 0
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.test: break 
        for step in tqdm(range(num_batch)): 
            u, seq, pos, neg = sampler.next_batch() 
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            
            pos_logits, neg_logits = model(seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            if args.alpha:
                for param in model.cate_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            
            loss.backward()
            optimizer.step()
            
        
        print("loss in epoch {}: {}".format(epoch, loss.item())) 
        
        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (HR@10: %0.4f, nDCG@10: %0.4f, ILD@10: %0.4f, cov@10: %0.4f)'
                    % (epoch, T, t_test[0][0], t_test[1][0], t_test[2][0], t_test[3][0]))
            f.write('%0.4f\t%0.4f\t%0.4f\t%0.4f'
                    % (t_test[0][0], t_test[1][0], t_test[2][0], t_test[3][0]) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
            
            if t_test[1][0] >= best:
                best = t_test[1][0]
                torch.save(model.state_dict(),  'runs/'+args.dataset + '/' + args.dir + '/model.pth')
                p = 0
            else:
                p += 1
            
            if p > 30:
                f.write('Done')
                break   
        
    model.load_state_dict(torch.load('runs/'+args.dataset + '/' + args.dir + '/model.pth', map_location=torch.device(args.device)))
    model.eval()
    t_test = evaluate(model, dataset, args, test_f=True, lamb=args.lamb)
    if args.lamb:
        print('\nlambda', args.lamb)
        f.write('\nlambda %s' % (args.lamb))

    print('\n----- TEST (N=[10,20]) ----- \n\n HR:\t%s \n nDCG:\t%s \n ILD:\t%s \n cov:\t%s \n' % (np.round(t_test[0], 4), np.round(t_test[1], 4), np.round(t_test[2], 4), np.round(t_test[3], 4)))
    f.write('\n----- TEST (N=[10,20]) ----- \n\n HR:\t%s \n nDCG:\t%s \n ILD:\t%s \n cov:\t%s \n' % (np.round(t_test[0], 4), np.round(t_test[1], 4), np.round(t_test[2], 4), np.round(t_test[3], 4)))
    f.close()
    sampler.close()
