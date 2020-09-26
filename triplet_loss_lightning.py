import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from rownet import RowNet
import random
from losses import triplet_loss_cosine_abext_marker

class triplet_loss(pl.LightningModule):
    def __init__(self, train_data, pos_neg_examples, learning_rate, embedded_dim=1024):
        # Language (BERT): 3072, Vision + Depth (ResNet152): 2048 * 2
        super(triplet_loss, self).__init__()
        self.language_train_data = [l for l, _, _, _ in train_data]
        self.vision_train_data = [v for _, v, _, _ in train_data]
        self.instance_names = [i for _, _, _, i in train_data]
        language_dim = list(self.language_train_data[0].size())[0]
        # Eitel dimension
        vision_dim = list(self.vision_train_data[0].size())[0]
        self.vision_model = RowNet(vision_dim, embedded_dim=embedded_dim)
        self.language_model = RowNet(language_dim, embedded_dim=embedded_dim)
        self.pos_neg_examples = pos_neg_examples
        self.learning_rate = learning_rate
        self.train_data = train_data
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=.2)
        x = self.fc3(x)

        return x

    def get_examples_batch(self, pos_neg_examples, indices, train_data, instance_names):
        examples = [pos_neg_examples[i] for i in indices]

        return (
            torch.stack([train_data[i[0]] for i in examples]),
            torch.stack([train_data[i[1]] for i in examples]),
            [instance_names[i[0]] for i in examples][0],
            [instance_names[i[1]] for i in examples][0],
        )
    def training_step(self, batch, batch_idx):
        speech, vision, object_name, instance_name = batch
        indices = list(range(batch_idx * len(batch), min((batch_idx + 1) * len(batch), len(self.train_data))))
        speech_pos, speech_neg, speech_pos_instance, speech_neg_instance = self.get_examples_batch(self.pos_neg_examples, indices,
                                                                                              self.speech_train_data,
                                                                                              self.instance_names)
        vision_pos, vision_neg, vision_pos_instance, vision_neg_instance = self.get_examples_batch(self.pos_neg_examples, indices,
                                                                                              self.vision_train_data,
                                                                                              self.instance_names)
        case = random.randint(1, 8)
        if case == 1:
            target = self.vision_model(vision)
            pos = self.vision_model(vision_pos)
            neg = self.vision_model(vision_neg)
            marker = ["bbb"]
        elif case == 2:
            target = self.speech_model(speech)
            pos = self.speech_model(speech_pos)
            neg = self.speech_model(speech_neg)
            marker = ["aaa"]
        elif case == 3:
            target = self.vision_model(vision)
            pos = self.speech_model(speech_pos)
            neg = self.speech_model(speech_neg)
            marker = ["baa"]
        elif case == 4:
            target = self.speech_model(speech)
            pos = self.vision_model(vision_pos)
            neg = self.vision_model(vision_neg)
            marker = ["abb"]
        elif case == 5:
            target = self.vision_model(vision)
            pos = self.vision_model(vision_pos)
            neg = self.speech_model(speech_neg)
            marker = ["bba"]
        elif case == 6:
            target = self.speech_model(speech)
            pos = self.speech_model(speech_pos)
            neg = self.vision_model(vision_neg)
            marker = ["aab"]
        elif case == 7:
            target = self.vision_model(vision)
            pos = self.speech_model(speech_pos)
            neg = self.vision_model(vision_neg)
            marker = ["bab"]
        elif case == 8:
            target = self.speech_model(speech)
            pos = self.vision_model(vision_pos)
            neg = self.speech_model(speech_neg)
            marker = ["aba"]
        loss = triplet_loss_cosine_abext_marker(target, pos, neg, marker, margin=0.4)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.language_model.parameters()) + list(self.vision_model.parameters()), lr=self.learning_rate)
        return optimizer