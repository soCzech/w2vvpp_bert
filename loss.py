import torch


def cosine_sim(query, retrio):
    query = torch.div(query, torch.pow(query, 2).sum(dim=1, keepdim=True).sqrt())
    retrio = torch.div(retrio, torch.pow(retrio, 2).sum(dim=1, keepdim=True).sqrt())
    return query.mm(retrio.t())


class ContrastiveLoss(torch.nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False, cost_style='sum', direction='bidir'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        self.max_violation = max_violation

    def forward(self, s, im):
        # compute image-sentence score matrix
        scores = cosine_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in ['i2t', 'bidir']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'bidir']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s = cost_s.max(1)[0]
            if cost_im is not None:
                cost_im = cost_im.max(0)[0]

        if cost_s is None:
            cost_s = torch.zeros(1).cuda()
        if cost_im is None:
            cost_im = torch.zeros(1).cuda()

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()
