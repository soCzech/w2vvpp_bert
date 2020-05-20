import torch


class W2VV3(torch.nn.Module):
    def __init__(self, train_backbone=True, bow_vect_len=None, reproject=False, normalize=False, use_w2v=False,
                 joint_space_dim=2048):
        super(W2VV3, self).__init__()
        self.use_bow = bow_vect_len is not None
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.train_backbone = train_backbone
        self.vis_transform = TransformNet(4096, joint_space_dim)
        self.txt_transform = TransformNet(
            (2048 if reproject else 768) + (bow_vect_len if self.use_bow else 0) + (500 if use_w2v else 0),
            joint_space_dim)

        self.relu = torch.nn.ReLU(inplace=True)
        self.reproject_layer = torch.nn.Linear(768, 2048) if reproject else None
        self.normalize = normalize

        if self.reproject_layer is not None:
            torch.nn.init.xavier_uniform_(self.reproject_layer.weight)
            torch.nn.init.zeros_(self.reproject_layer.bias)

    def encode_visual(self, vis_vect):
        return self.vis_transform(vis_vect)

    def encode_text(self, txt_vect, txt_lens, bow_vect=None):
        if self.train_backbone:
            txt_vect = self.roberta.extract_features(txt_vect)
        else:
            with torch.no_grad():
                txt_vect = self.roberta.extract_features(txt_vect)

        if self.reproject_layer is not None:
            txt_vect = self.relu(txt_vect)
            txt_vect = self.reproject_layer(txt_vect)

        # remove <start> token
        txt_vect = txt_vect[:, 1:]
        # remove <end> token
        txt_lens = txt_lens - 2

        batch_size, max_len, n_channels = txt_vect.shape
        idx = torch.arange(max_len).cuda().unsqueeze(0).expand(batch_size, -1)
        idx = idx < txt_lens.unsqueeze(1)
        idx = idx.unsqueeze(2).expand(-1, -1, n_channels)
        txt_vect = (txt_vect * idx.float()).sum(1) / txt_lens.unsqueeze(1).float()

        if self.normalize:
            txt_vect = torch.nn.functional.normalize(txt_vect, dim=-1, p=2)

        if self.use_bow:
            return self.txt_transform(txt_vect, bow_vect)
        return self.txt_transform(txt_vect)


class TransformNet(torch.nn.Module):
    def __init__(self, input_dim, joint_space_dim=2048):
        super(TransformNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, joint_space_dim)
        self.activation = torch.nn.Tanh()

        self.dropout = torch.nn.Dropout(p=0.2)
        self.init_weights()

    def init_weights(self):
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, vector, vector2=None):
        if vector2 is not None:
            vector = torch.cat([vector, vector2], 1)
        features = self.fc1(vector)
        features = self.activation(features)
        features = self.dropout(features)
        return features
