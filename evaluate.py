import tqdm
import torch
import argparse
import numpy as np

from model import W2VV3
from dataset import TextDataset, VisionDataset
from encoder import Text2BoWEncoder, Text2W2VEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--model_weights", type=str, required=True, help="path to the model checkpoint file")
parser.add_argument("--output_file", type=str, required=True, help="path and name of the output txt file")
parser.add_argument("--vision_ds", type=str, required=True, help="path to vision dataset directory")
parser.add_argument("--text_ds", type=str, required=True, help="path to text query file")
parser.add_argument("--w2v_weights", type=str, required=True, help="path to w2v weights file")
parser.add_argument("--bow_vocab", type=str, required=True, help="path to bow vocabulary text file")

args = parser.parse_args()


"""
INITIALIZE MODEL
"""

net = W2VV3(bow_vect_len=11147, use_w2v=True).cuda().eval()

checkpoint = torch.load(args.model_weights)
net.load_state_dict(checkpoint["model_state_dict"])


"""
ENCODE FRAMES
"""

vis_ds = VisionDataset(args.vision_ds)

vis_ids = []
vis_emb = np.empty([len(vis_ds), 2048])
for i in tqdm.trange(len(vis_ds)):
    id_str, tensor = vis_ds[i]

    with torch.no_grad():
        tensor = tensor.view([1, -1]).cuda()
        tensor = net.encode_visual(tensor)
        tensor = tensor.cpu().numpy()[0]

    vis_ids.append(id_str)
    vis_emb[i] = tensor


"""
ENCODE TEXT QUERIES
"""

txt_ds = TextDataset(args.text_ds)
bow_encoder = Text2BoWEncoder(args.bow_vocab)
w2v_encoder = Text2W2VEncoder(args.w2v_weights)

txt_ids = []
txt_emb = np.empty([len(txt_ds), 2048])
for i in tqdm.trange(len(txt_ds)):
    id_str, tensor = txt_ds[i]
    bow_tensor = bow_encoder.encode(tensor)
    w2v_tensor = w2v_encoder.encode(tensor)
    roberta_tensor = net.roberta.encode(tensor)

    with torch.no_grad():
        roberta_tensor_len = torch.LongTensor([len(roberta_tensor)]).cuda()
        roberta_tensor = roberta_tensor.view([1, -1]).cuda()
        static_tensor = torch.cat([bow_tensor, w2v_tensor], 0).view([1, -1]).cuda()

        tensor = net.encode_text(roberta_tensor, roberta_tensor_len, static_tensor)
        tensor = tensor.cpu().numpy()[0]

    txt_ids.append(id_str)
    txt_emb[i] = tensor


"""
COMPUTE SIMILARITIES
"""

txt_emb = txt_emb / np.linalg.norm(txt_emb, axis=1, keepdims=True)
vis_emb = vis_emb / np.linalg.norm(vis_emb, axis=1, keepdims=True)
sim_matrix = txt_emb.dot(vis_emb.T)
sorted_result_lists = np.argsort(-sim_matrix, axis=1)  # minus in order to get the highest first


"""
SAVE RESULTS
"""

with open(args.output_file, "w") as f:
    for query_idx, result_list in enumerate(sorted_result_lists):
        txt_id = txt_ids[query_idx]
        vis_id = [f"{vis_ids[res_idx]} {sim_matrix[query_idx][res_idx]:.5f}" for res_idx in result_list]
        f.write(f"{txt_id} {' '.join(vis_id)}\n")
