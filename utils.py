import numpy as np


def cosine_sim(query_embs, retro_embs):
    query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    retro_embs = retro_embs / np.linalg.norm(retro_embs, axis=1, keepdims=True)

    return query_embs.dot(retro_embs.T)


def get_label_matrix(vis_emb, vis_ids, txt_emb, txt_ids):
    unique_vis_ids, unique_vis_ids_idx = [], []
    for i, vid in enumerate(vis_ids):
        if vid not in unique_vis_ids:
            unique_vis_ids.append(vid)
            unique_vis_ids_idx.append(i)
    vis_ids = unique_vis_ids
    vis_emb = vis_emb[unique_vis_ids_idx]

    txt2vis_sim = cosine_sim(txt_emb, vis_emb)
    inds = np.argsort(txt2vis_sim, axis=1)
    label_matrix = np.zeros(inds.shape)
    for index in range(inds.shape[0]):
        ind = inds[index][::-1]
        label_matrix[index][np.where(np.array(vis_ids)[ind] == txt_ids[index].split('#')[0])[0]] = 1
    return label_matrix


def compute_metrics(label_matrix):
    ranks = np.zeros(label_matrix.shape[0])
    aps = np.zeros(label_matrix.shape[0])

    for index in range(len(ranks)):
        rank = np.where(label_matrix[index] == 1)[0] + 1
        ranks[index] = rank[0]

        aps[index] = np.mean([(i+1.)/rank[i] for i in range(len(rank))])

    r1, r5, r10 = [100.0*np.mean([x <= k for x in ranks]) for k in [1, 5, 10]]
    medr = np.floor(np.median(ranks))
    meanr = ranks.mean()
    mir = (1.0/ranks).mean()
    mAP = aps.mean()

    return r1, r5, r10, medr, meanr, mir, mAP
