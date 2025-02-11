import torch
import numpy as np
from tqdm import tqdm
from metrics import ndcg_and_recall

def train(data_loader, model, optimizer, device):

    epoch_loss = 0
    epoch_other_information = []
    data_loader.dataset.row_neg_sampling()  #采负样本
    for i, batch in enumerate(tqdm(data_loader)):  #batch
      if i<3:  #为了验证运行无误
        row_ids, pos, neg = batch
        row_ids = row_ids.long().cuda(torch.device(device))
        pos = pos.long().cuda(torch.device(device))
        neg = neg.long().cuda(torch.device(device))
        col_ids = torch.concat([pos, neg], dim=0)

        # feed
        optimizer.zero_grad()
        loss, other_information= model(data_loader.dataset.adj_norm, row_ids, col_ids, pos, neg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.cpu().item()
        epoch_other_information.append(other_information.cpu().data)

        torch.cuda.empty_cache() #清空缓存

    #计算每个batch的平均损失
    batch_number = len(data_loader)
    epoch_loss = epoch_loss / batch_number
    epoch_other_information = torch.sum(torch.stack(epoch_other_information), dim=0) / batch_number

    return epoch_loss, epoch_other_information


def test(data_loader, model, test_batch_size,topk,device):

        test_row_ids = np.array([i for i in range(data_loader.dataset.row_number)])
        batch_number = int(np.ceil(len(test_row_ids) / test_batch_size))

        all_recall= 0
        all_ndcg = 0
        for batch in tqdm(range(batch_number)):
          if batch<1 :
            start = batch * test_batch_size
            end = min((batch+1) * test_batch_size, len(test_row_ids))  #可能会超过因此取最小

            #测试的用户
            row_ids = torch.LongTensor(test_row_ids[start:end]).cuda(torch.device(device))
            E_r,E_c = model.get_embedding()
            preds = E_r[row_ids] @ E_c.T
            mask = data_loader.dataset.train_interaction[row_ids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(device))
            preds = preds * (1 - mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)  # 排序

            #top@ x
            recall, ndcg = ndcg_and_recall(test_row_ids[start:end], predictions, topk, data_loader.dataset.test_labels)

            all_recall += recall
            all_ndcg += ndcg

        return all_recall,all_ndcg