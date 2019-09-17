def remove_bert_heads(model, layer_idx, heads):
    model.encoder.layer[layer_idx].attention.prune_heads(heads)