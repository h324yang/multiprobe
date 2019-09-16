import argparse

from pytorch_transformers import BertForMaskedLM, BertTokenizer
import torch

from multiprobe.model import SingleInputBundle, predict_top_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert-model', type=str, default='bert-base-multilingual-uncased')
    args = parser.parse_args()

    model = BertForMaskedLM.from_pretrained(args.bert_model).cuda()
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    while True:
        with torch.no_grad():
            sentence = input('> ')
            bundle = SingleInputBundle([tokenizer.tokenize(sentence)], tokenizer.vocab)
            bundle.cuda()
            print(predict_top_k(model, tokenizer.vocab, tokenizer.ids_to_tokens, bundle))


if __name__ == '__main__':
    main()
