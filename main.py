import torch
from model import get_model
import torch.utils.data as data
from data_process import data_loader_hnd_handle
import hyperParameter
from train_and_test import train,test
from result_process import result_process,get_best_result



#导入超参数
args = hyperParameter.parse_args()
device =f'cuda:{args.cuda}'

# load data
data_container = data_loader_hnd_handle(args.dataset_name, device)

data_loader = data.DataLoader(data_container, batch_size=args.train_batch_size, shuffle=True, num_workers=0)

#创建模型
model = get_model(args.model_name)(args, data_container.row_number, data_container.col_number, device)
model.cuda(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=args.lr)
#optimizer.load_state_dict(torch.load('saved_optim.pt'))

#记录结果
all_recall = {}
all_ndcg = {}
for topk in args.topks:
    all_recall[f'recall@{topk}']=[]
    all_ndcg[f'ndcg@{topk}'] = []

train_epoch_index = []
test_epoch_index = []
loss = []
other_information = []
current_best_result={}
for topk  in args.topks:
    current_best_result[f"current_best_ndcg{topk}"]=0
    current_best_result[f"current_best_recall{topk}"] = 0
    current_best_result[f"current_best_recall{topk}_weight"]=None


for epoch in range(args.epoch):  #epoch

    if epoch<3: # 为验证无误设置，正式实验去除
        epoch_loss,epoch_other_information = train(data_loader,model,optimizer,device)
        train_epoch_index.append(epoch)
        loss.append(epoch_loss)
        other_information.append(epoch_other_information)

        if epoch % 1 == 0:  # test every 10 epochs
            test_epoch_index.append(epoch)
            for topk in args.topks:
                  epoch_recall, epoch_ndcg = test(data_loader, model, args.test_batch_size, topk, device)
                  all_recall[f'recall@{topk}'].append(epoch_recall)
                  all_ndcg[f'ndcg@{topk}'].append(epoch_ndcg)

                  if current_best_result[f"current_best_ndcg{topk}"]<=epoch_ndcg and current_best_result[f"current_best_recall{topk}"]<=epoch_recall:
                        current_best_result[f"current_best_recall{topk}"] = epoch_ndcg
                        current_best_result[f"current_best_ndcg{topk}"] = epoch_recall
                        current_best_result[f"current_best_{topk}_state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}


        # print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)


#处理和存储结果
result_process(args,train_epoch_index,loss,other_information,test_epoch_index,all_recall,all_ndcg,current_best_result)
get_best_result(args,f'./result/{args.model_name}/all_results')