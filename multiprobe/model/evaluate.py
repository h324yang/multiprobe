from pytorch_transformers import BertForMaskedLM
import torch


def hook_bert_layer_output(bert_model, layer_idx, hook):
    bert_model.encoder.layer[layer_idx].register_forward_hook(hook)


def hook_bert_layer_attn(bert_model, layer_idx, hook):
    bert_model.encoder.layer[layer_idx].attention.self.register_forward_hook(hook)


def hook_bert_word_embeddings(bert_model, hook):
    bert_model.embeddings.word_embeddings.register_forward_hook(hook)


class SingleInputBundle(object):

    def __init__(self, sentences, id_map, max_len=256):
        sentences = [x[:max_len - 2] for x in sentences]
        self.sentences = [['[CLS]'] + x + ['[SEP]'] for x in sentences]
        self.padded_sentences = [x + ['[PAD]'] * (max_len - len(x)) for x in self.sentences]
        self.token_ids = torch.tensor([list(map(id_map.__getitem__, x)) for x in self.padded_sentences])
        self.input_mask = torch.tensor([([1] * len(x)) + ([0] * (max_len - len(x))) for x in self.sentences])
        self.segment_ids = torch.tensor([[0] * max_len for _ in sentences])

    def cuda(self):
        self.token_ids = self.token_ids.cuda()
        self.input_mask = self.input_mask.cuda()
        self.segment_ids = self.segment_ids.cuda()
        return self

    def mean(self, tensor):
        mask = self.input_mask.unsqueeze(-1).expand_as(tensor).float()
        return (tensor * mask).sum(1) / mask.sum(1)


def predict_top_k(model: BertForMaskedLM, encode_map, decode_map, input_bundle, k=10):
    mask_id = encode_map['[MASK]']
    scores, = model(input_bundle.token_ids, input_bundle.segment_ids, input_bundle.input_mask)
    scores_mask = torch.tensor([[x == '[MASK]' for x in sentence] for sentence in input_bundle.padded_sentences])
    predictions = []
    for score_slice, mask in zip(scores, scores_mask):
        _, indices = torch.topk(score_slice[mask], k)
        for indices_slice in indices:
            predictions.append(list(map(decode_map.__getitem__, indices_slice.cpu().tolist())))
    return predictions
