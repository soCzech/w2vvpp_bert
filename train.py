import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime


from model import W2VV3
from loss import ContrastiveLoss
from dataset import PairDataset, RobertaDataset, roberta_collate_tokens
from utils import compute_metrics, get_label_matrix


# Settings
###########
run_name = datetime.strftime(datetime.now(), "%Y-%m-%d_%H%M%S")
# general setting
no_epochs = 30
train_backbone = True
joint_space_dim = 2048
batch_size = 96 if train_backbone else 128  # due to gpu memory
# bow
bow_vect_len = 11147  # None if trained without bow
vocab_file = "./data/bow_nsw_5.txt" if bow_vect_len is not None else None
# word2vec
use_w2v = True
w2v_file = "./data/word2vec/flickr/vec500flickr30m" if use_w2v else None


# Network definition
#####################
criterion = ContrastiveLoss(margin=0.2, max_violation=True, direction="t2i")  # NCLoss()

net = W2VV3(train_backbone=train_backbone,
            bow_vect_len=bow_vect_len,
            use_w2v=use_w2v,
            joint_space_dim=joint_space_dim).cuda()


# Optimizer
############
if train_backbone:
    optimizer = torch.optim.Adam(net.parameters(), betas=(0.9, 0.98), eps=1e-06, weight_decay=0., lr=1e-05)
else:
    optimizer = torch.optim.RMSprop(
        list(net.vis_transform.parameters()) + list(net.txt_transform.parameters()), lr=0.0001)


# Datasets
###########
ds = PairDataset(
    "./data/tgif-msrvtt10k/FeatureData/mean_resnext101_resnet152/",
    "./data/tgif-msrvtt10k/TextData/tgif-msrvtt10k.caption.txt"
)
ds = RobertaDataset(ds, net.roberta.encode, bow_vocab_file=vocab_file, w2v_file=w2v_file)
data_loader = torch.utils.data.DataLoader(
    dataset=ds,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=False,
    num_workers=4,
    collate_fn=roberta_collate_tokens)

val_ds = PairDataset(
    "./data/tv2016train/FeatureData/mean_resnext101_resnet152/",
    "./data/tv2016train/TextData/setA/tv2016train.caption.txt"
)
val_data_loader = RobertaDataset(val_ds, net.roberta.encode, bow_vocab_file=vocab_file, w2v_file=w2v_file)
val_data_loader = torch.utils.data.DataLoader(
    dataset=val_data_loader,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=False,
    num_workers=4,
    collate_fn=roberta_collate_tokens)


# Training
###########
last_best_checkpoint, last_best_value = None, 0

for epoch in range(1, no_epochs + 1):
    print("Epoch {}:".format(epoch))
    net.train()
    optimizer.zero_grad()

    loss_per_batch = []
    for vis_inputs, txt_inputs, txt_lens, bow_inputs in tqdm(data_loader):
        txt_outputs = net.encode_text(txt_inputs.cuda(), txt_lens.cuda(),
                                      bow_inputs.cuda() if bow_inputs is not None else None)
        vis_outputs = net.encode_visual(vis_inputs.cuda())

        loss = criterion(s=txt_outputs, im=vis_outputs)
        loss.backward()
        loss_per_batch.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()

    loss_val = sum(loss_per_batch) / len(loss_per_batch)
    print(f"\tLoss: {loss_val:.4f}")

    # EVAL
    net.eval()
    idx = 0
    vis_emb = np.empty([len(val_ds), joint_space_dim])
    txt_emb = np.empty([len(val_ds), joint_space_dim])
    txt_ids, vis_ids = [], []

    for vis_inputs, txt_inputs, txt_lens, bow_inputs in tqdm(val_data_loader):
        with torch.no_grad():
            txt_outputs = net.encode_text(txt_inputs.cuda(), txt_lens.cuda(),
                                          bow_inputs.cuda() if bow_inputs is not None else None).cpu().numpy()
            vis_outputs = net.encode_visual(vis_inputs.cuda()).cpu().numpy()

        for i in range(len(txt_outputs)):
            txt_emb[idx] = txt_outputs[i]
            vis_emb[idx] = vis_outputs[i]

            cap_id = val_ds.cap_ids[idx]
            vis_id = cap_id.split('#', 1)[0]
            txt_ids.append(cap_id)
            vis_ids.append(vis_id)
            idx += 1

    label_matrix = get_label_matrix(vis_emb, vis_ids, txt_emb, txt_ids)
    r1, r5, r10, medr, meanr, mir, mAP = compute_metrics(label_matrix)

    print(" * Text to video:")
    print(" * r_1_5_10: {}".format([round(r1, 3), round(r5, 3), round(r10, 3)]))
    print(" * medr, meanr, mir: {}".format([round(medr, 3), round(meanr, 3), round(mir, 3)]))
    print(" * mAP: {}".format(round(mAP, 3)))
    print(" * " + '-' * 10)
    if mAP > last_best_value:
        os.makedirs("./bert_models", exist_ok=True)
        checkpoint_name = "./bert_models/{}-epoch{:02}-loss{:.4f}.pth.tar".format(run_name, epoch, loss_val)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'loss': loss_val,
        }, checkpoint_name)

        if last_best_checkpoint is not None:
            os.remove(last_best_checkpoint)

        last_best_value = mAP
        last_best_checkpoint = checkpoint_name
