import torch
from bigfile import BigFile


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
