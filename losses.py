import torch

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
