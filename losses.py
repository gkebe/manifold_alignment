import torch
import torch.nn.functional as F

def cosine_pairwise_loss(d_a, d_b, match, model_A, model_B):
    criterion = torch.nn.CosineEmbeddingLoss()
    loss = criterion(model_A(d_a),model_B(d_b),torch.tensor(match))
    return loss

def triplet_loss_vanilla(anchor, positive, negative, language_model, vision_model, margin=0.4):
    triplet_loss = torch.nn.TripletMarginLoss(margin=margin)
    loss = triplet_loss(
        vision_model(anchor),
        language_model(positive),
        language_model(negative)
    )

    return loss

def cosine_triplet_loss(margin=0.4):
    #cosdist = lambda x1, x2: 1 - F.cosine_similarity(x1, x2)
    #lossf = lambda a, p, n: torch.clamp(cosdist(a,p) - cosdist(a,n) + margin, 0.0, 2.0 + margin)
    lossf = lambda a, p, n: torch.clamp(F.cosine_similarity(a, n) - F.cosine_similarity(a, p) + margin, 0.0, 2.0 + margin)
    return lossf


def triplet_loss_cosine_abext_marker(anchor, positive, negative, marker, margin=0.4):
    """Triplet cosine loss with intra-class splitting."""
    triplet_loss = cosine_triplet_loss(margin=margin)
    marker_types = ['aaa', 'aab', 'aba', 'baa', 'bba', 'bab', 'abb', 'bbb']
    marker_dict = {k: [i for i in range(len(marker)) if marker[i] == k] for k in marker_types}

    l = 0.0
    num = 0

    for marker_type in marker_types:
        if len(marker_dict[marker_type]) == 0:
            continue
        else:
            num = num + len(marker_dict[marker_type])

            l = l + torch.sum(triplet_loss(
                anchor[marker_dict[marker_type]],
                positive[marker_dict[marker_type]],
                negative[marker_dict[marker_type]]
            ))

    return l / num