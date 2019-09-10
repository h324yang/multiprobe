import argparse

from pytorch_transformers import BertModel, BertTokenizer
from tqdm import tqdm
import pandas as pd
import torch

from multiprobe.utils import chunk
from multiprobe.model import SingleInputBundle, hook_bert


class BundleAveragingHook(object):

    def __init__(self):
        self.bundle = None
        self.data = []

    def __call__(self, module, tensor_in, tensor_out):
        if not self.bundle:
            return
        self.data.append(self.bundle.mean(tensor_out[0].detach()).cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file-prefix', '-o', type=str, required=True)
    parser.add_argument('--dataset-file', type=str, required=True)
    parser.add_argument('--bert-model', type=str, default='bert-base-multilingual-uncased')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--layers', type=int, nargs='+', default=[-1, 0, 5])
    args = parser.parse_args()

    model = BertModel.from_pretrained(args.bert_model).cuda()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    hooks = [BundleAveragingHook() for _ in args.layers]
    for layer_idx, hook in zip(args.layers, hooks):
        hook_bert(model, layer_idx, hook)
    df = pd.read_csv(args.dataset_file, sep='\t', quoting=3)
    for data in tqdm(list(chunk(list(df.itertuples()), args.batch_size))):
        _, languages, sentences = list(zip(*data))
        bundle = SingleInputBundle(list(map(str.split, sentences)), tokenizer.vocab)
        bundle.cuda()
        for hook in hooks:
            hook.bundle = bundle
        model(bundle.token_ids, bundle.segment_ids, bundle.input_mask)
    for layer_idx, hook in zip(args.layers, hooks):
        torch.save(torch.cat(hook.data), f'{args.output_file_prefix}-l{layer_idx}.pt')


if __name__ == '__main__':
    main()