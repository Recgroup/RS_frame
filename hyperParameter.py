import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--dataset_name', default='yelp', type=str, help='dataset name')
    parser.add_argument('--model_name', default='LightGCL', type=str, help='modelName')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    parser.add_argument('--test_batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--train_batch_size', default=4096, type=int, help='batch size')
    parser.add_argument('--topks', default=[20,40], type=list, help='topk')
    parser.add_argument('--lambda1', default=0.2, type=float, help='weight of cl loss')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--dim', default=64, type=int, help='embedding size')
    parser.add_argument('--svd_q', default=5, type=int, help='rank')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--dropout', default=0.0, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
    parser.add_argument('--lambda2', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')
    return parser.parse_args()