import torch
from bigfile import BigFile

from encoder import Text2W2VEncoder, Text2BoWEncoder


class VisionDataset(torch.utils.data.Dataset):

    def __init__(self, filename):
        self.vis_feat_file = BigFile(filename)
        self.vis_ids = self.vis_feat_file.names

    def __getitem__(self, index):
        vis_tensor = self.vis_feat_file.read_one(self.vis_ids[index])
        return self.vis_ids[index], torch.Tensor(vis_tensor)

    def get_by_name(self, name):
        vis_tensor = self.vis_feat_file.read_one(name)
        return torch.Tensor(vis_tensor)

    def __len__(self):
        return len(self.vis_ids)


class TextDataset(torch.utils.data.Dataset):

    def __init__(self, filename):
        self.captions = {}
        self.cap_ids = []
        with open(filename, "r", encoding="utf-8") as reader:
            for line in reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)

    def __getitem__(self, index):
        return self.cap_ids[index], self.captions[self.cap_ids[index]]

    def get_by_name(self, name):
        return self.captions[name]

    def __len__(self):
        return len(self.cap_ids)


class PairDataset(torch.utils.data.Dataset):

    def __init__(self, visual_filename, text_filename):
        self.vis_data = VisionDataset(visual_filename)
        self.txt_data = TextDataset(text_filename)
        self.cap_ids = self.txt_data.cap_ids

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        vis_id = cap_id.split('#', 1)[0]
        return self.vis_data.get_by_name(vis_id), self.txt_data.get_by_name(cap_id)

    def __len__(self):
        return len(self.cap_ids)


class RobertaDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, encode_fc, bow_vocab_file=None, w2v_file=None):
        self.original_dataset = dataset
        self.encode_fc = encode_fc
        self.bow_encoder = Text2BoWEncoder(bow_vocab_file) if bow_vocab_file is not None else None
        self.w2v_encoder = Text2W2VEncoder(w2v_file) if w2v_file is not None else None

    def __getitem__(self, index):
        vis_vect, txt_vect = self.original_dataset[index]
        txt_vect_encoded = self.encode_fc(txt_vect)

        txt_bow_vect = None
        if self.bow_encoder is not None:
            txt_bow_vect = self.bow_encoder.encode(txt_vect)
        if self.w2v_encoder is not None:
            txt_w2v_vect = self.w2v_encoder.encode(txt_vect)
            txt_bow_vect = torch.cat([txt_bow_vect, txt_w2v_vect], 0) if txt_bow_vect is not None else txt_w2v_vect

        return vis_vect, txt_vect_encoded, len(txt_vect_encoded), txt_bow_vect

    def __len__(self):
        return len(self.original_dataset)


def roberta_collate_tokens(values, pad_idx=1):
    vis_values = torch.stack([v for v, t, l, bow in values])
    txt_values = [t for v, t, l, bow in values]
    txt_lens = [l for v, t, l, bow in values]
    bow_values = torch.stack([bow for v, t, l, bow in values]) if values[0][3] is not None else None

    size = max(v.size(0) for v in txt_values)
    res = txt_values[0].new(len(txt_values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(txt_values):
        copy_tensor(v, res[i][:len(v)])
    return vis_values, res, torch.LongTensor(txt_lens), bow_values
