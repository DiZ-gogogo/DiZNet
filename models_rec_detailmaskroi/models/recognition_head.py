import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from .losses import acc
import numpy as np
from .vec2word_pyx import vec2word


class RecognitionHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, voc, char2id, id2char,
                 feature_size=(8, 32)):
        super(RecognitionHead, self).__init__()
        self.char2id = char2id
        _id2char = []
        for i in range(max(id2char.keys()) + 1):
            if i in id2char:
                _id2char.append(id2char[i])
            else:
                _id2char.append('PAD')
        _id2char = np.array(_id2char)
        id2char = _id2char
        self.id2char = id2char

        self.conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1,
                              padding=1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.feature_size = feature_size
        self.encoder = Encoder(hidden_dim, voc, char2id, id2char)
        self.decoder = Decoder(hidden_dim, hidden_dim, 2, voc, char2id, id2char)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, output_size):
        return F.upsample(x, size=output_size, mode='bilinear')

    def extract_feature(self, f, output_size, instance, bboxes, gt_words=None,
                        word_masks=None, unique_labels=None, feature_kernel=None):

        # print("f rec inside:", f.shape)
        # print("f kernel inside", feature_kernel.shape)

        fk = feature_kernel.squeeze()


        instance = fk.long() + instance.long()



        x = self.conv(f)
        x = self.relu(self.bn(x))
        x = self._upsample(x, output_size)

        # print("fk  inside", fk.shape)
        #
        # print("fk + instance  inside", instance.shape)
        # print(instance)
        #
        # print("f feature ext", x.shape)
        # print("instance", instance.shape)
        # print("word mask:", word_masks.shape)
        # print("-------------------------------------")

        x_crops = []
        if gt_words is not None:
            words = []

        batch_size, _, H, W = x.size()
        pad_scale = 1
        pad = x.new_tensor([-1, -1, 1, 1], dtype=torch.long) * pad_scale
        if self.training:
            offset = x.new_tensor(
                np.random.randint(-pad_scale, pad_scale + 1, bboxes.size()),
                dtype=torch.long)
            pad = pad + offset

        bboxes = bboxes + pad
        bboxes[:, :, (0, 2)] = bboxes[:, :, (0, 2)].clamp(0, H)
        bboxes[:, :, (1, 3)] = bboxes[:, :, (1, 3)].clamp(0, W)

        for i in range(x.size(0)):
            instance_ = instance[i:i + 1]
            if unique_labels is None:
                unique_labels_, _ = torch.unique(instance_, sorted=True,
                                                 return_inverse=True)
            else:
                unique_labels_ = unique_labels[i]
            x_ = x[i]
            if gt_words is not None:
                gt_words_ = gt_words[i]
            if word_masks is not None:
                word_masks_ = word_masks[i]
            bboxes_ = bboxes[i]

            for label in unique_labels_:
                if label == 0:
                    continue
                if word_masks is not None and word_masks_[label] == 0:
                    continue
                t, l, b, r = bboxes_[label]

                mask = (instance_[:, t:b, l:r] == label).float()
                mask = \
                    F.max_pool2d(mask.unsqueeze(0), kernel_size=(3, 3), stride=1,
                                 padding=1)[0]

                if torch.sum(mask) == 0:
                    continue
                x_crop = x_[:, t:b, l:r] * mask
                _, h, w = x_crop.size()
                if h > w * 1.5:
                    x_crop = x_crop.transpose(1, 2)
                x_crop = F.interpolate(x_crop.unsqueeze(0), self.feature_size,
                                       mode='bilinear')
                x_crops.append(x_crop)
                if gt_words is not None:
                    words.append(gt_words_[label])
        if len(x_crops) == 0:
            return None, None
        x_crops = torch.cat(x_crops)
        if gt_words is not None:
            words = torch.stack(words)
        else:
            words = None
        return x_crops, words

    def extract_feature_test(self, f, output_size, instance, bboxes,
                             unique_labels):
        x = self.conv(f)
        x = self.relu(self.bn(x))
        x = self._upsample(x, output_size)

        x_crops = []

        batch_size, _, H, W = x.size()
        pad_scale = 1
        pad = x.new_tensor([-1, -1, 1, 1], dtype=torch.long) * pad_scale
        bboxes = bboxes + pad
        bboxes[:, :, (0, 2)] = bboxes[:, :, (0, 2)].clamp(0, H)
        bboxes[:, :, (1, 3)] = bboxes[:, :, (1, 3)].clamp(0, W)

        for i in range(x.size(0)):
            instance_ = instance[i:i + 1]
            unique_labels_ = unique_labels[i]
            x_ = x[i]
            bboxes_ = bboxes[i]

            for label in unique_labels_:
                if label == 0:
                    continue
                t, l, b, r = bboxes_[label]

                mask = (instance_[:, t:b, l:r] == label).float()
                mask = \
                    F.max_pool2d(mask.unsqueeze(0), kernel_size=(3, 3), stride=1,
                                 padding=1)[0]

                x_crop = x_[:, t:b, l:r] * mask
                _, h, w = x_crop.size()
                if h > w * 1.5:
                    x_crop = x_crop.transpose(1, 2)
                x_crop = F.interpolate(x_crop.unsqueeze(0), self.feature_size,
                                       mode='bilinear')
                x_crops.append(x_crop)

        if len(x_crops) == 0:
            return None
        x_crops = torch.cat(x_crops)

        return x_crops

    def loss(self, input, target, reduce=True):
        EPS = 1e-6
        N, L, D = input.size()
        mask = target != self.char2id['PAD']
        input = input.contiguous().view(-1, D)
        target = target.contiguous().view(-1)
        loss_rec = F.cross_entropy(input, target, reduce=False)
        loss_rec = loss_rec.view(N, L)
        loss_rec = torch.sum(loss_rec * mask.float(), dim=1) / (
                torch.sum(mask.float(), dim=1) + EPS)
        acc_rec = acc(torch.argmax(input, dim=1).view(N, L), target.view(N, L),
                      mask, reduce=False)
        if reduce:
            loss_rec = torch.mean(loss_rec[valid])
            acc_rec = torch.mean(acc_rec)
        losses = {'loss_rec': loss_rec, 'acc_rec': acc_rec}

        return losses

    def forward(self, x, target=None, args=None):
        holistic_feature = self.encoder(x)


        print("rec head",holistic_feature.shape)
        print("_____________________________")

        if self.training:
            return self.decoder(x, holistic_feature, target)
        else:
            return self.decoder.forward_test(x, holistic_feature)


class Encoder(nn.Module):

    def __init__(self, hidden_dim, voc, char2id, id2char):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = len(voc)
        self.START_TOKEN = char2id['EOS']
        self.emb = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.att = MultiHeadAttentionLayer(self.hidden_dim, 8)
        # self.att = MultiHeadAttentionLayer_Fast(self.hidden_dim, 8)

    def forward(self, x):
        batch_size, feature_dim, H, W = x.size()
        x_flatten = x.view(batch_size, feature_dim, H * W).permute(0, 2, 1)
        st = x.new_full((batch_size,), self.START_TOKEN, dtype=torch.long)
        emb_st = self.emb(st)
        holistic_feature, _ = self.att(emb_st, x_flatten, x_flatten)
        return holistic_feature


class Decoder(nn.Module):

    def __init__(self, featrue_dim, hidden_dim, num_layers, voc, char2id,
                 id2char):
        super(Decoder, self).__init__()
        self.featrue_dim = featrue_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = len(voc)
        self.START_TOKEN = char2id['EOS']
        self.END_TOKEN = char2id['EOS']
        self.NULL_TOKEN = char2id['PAD']
        self.id2char = id2char
        self.lstm_u = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm_u.append(nn.LSTMCell(
                self.hidden_dim, self.hidden_dim))
        self.emb = nn.Embedding(self.vocab_size, self.hidden_dim)

        self.att = MultiHeadAttentionLayer(self.hidden_dim, 8)
        # self.att = MultiHeadAttentionLayer_Fast(self.hidden_dim, 8)
        self.cls = nn.Linear(self.hidden_dim + self.featrue_dim,
                             self.vocab_size)

    def forward(self, x, holistic_feature, target):
        # print(x.shape, holistic_feature.shape, target.shape)
        batch_size, feature_dim, H, W = x.size()
        x_flatten = x.view(batch_size, feature_dim, H * W).permute(0, 2, 1)

        max_seq_len = target.size(1)
        h = []
        for i in range(self.num_layers):
            h.append(
                (x.new_zeros((x.size(0), self.hidden_dim), dtype=torch.float32),
                 x.new_zeros((x.size(0), self.hidden_dim),
                             dtype=torch.float32)))

        out = x.new_zeros((x.size(0), max_seq_len + 1, self.vocab_size),
                          dtype=torch.float32)
        for t in range(max_seq_len + 1):
            if t == 0:
                xt = holistic_feature
            elif t == 1:
                it = x.new_full((batch_size,), self.START_TOKEN,
                                dtype=torch.long)
                xt = self.emb(it)
            else:
                it = target[:, t - 2]
                xt = self.emb(it)

            for i in range(self.num_layers):
                if i == 0:
                    inp = xt
                else:
                    inp = h[i - 1][0]
                h[i] = self.lstm_u[i](inp, h[i])
            ht = h[-1][0]
            out_t, _ = self.att(ht, x_flatten, x_flatten)
            # print(out_t.shape, _.shape)
            out_t = torch.cat((out_t, ht), dim=1)
            # print(out_t.shape)
            # exit()
            out_t = self.cls(out_t)
            out[:, t, :] = out_t
        return out[:, 1:, :]

    def to_words(self, seqs, seq_scores=None):
        EPS = 1e-6
        words = []
        word_scores = None
        if seq_scores is not None:
            word_scores = []

        for i in range(len(seqs)):
            word = ''
            word_score = 0
            for j, char_id in enumerate(seqs[i]):
                char_id = int(char_id)
                if char_id == self.END_TOKEN:
                    break
                if self.id2char[char_id] in ['PAD', 'UNK']:
                    continue
                word += self.id2char[char_id]
                if seq_scores is not None:
                    word_score += seq_scores[i, j]
            words.append(word)
            if seq_scores is not None:
                word_scores.append(word_score / (len(word) + EPS))
        return words, word_scores

    def forward_test(self, x, holistic_feature):
        batch_size, feature_dim, H, W = x.size()
        x_flatten = x.view(batch_size, feature_dim, H * W).permute(0, 2, 1)

        h = x.new_zeros(self.num_layers, 2, batch_size, self.hidden_dim)

        max_seq_len = 32
        seq = x.new_full((batch_size, max_seq_len + 1), self.START_TOKEN,
                         dtype=torch.long)
        seq_score = x.new_zeros((batch_size, max_seq_len + 1),
                                dtype=torch.float32)
        end = x.new_ones((batch_size,), dtype=torch.uint8)
        for t in range(max_seq_len + 1):
            if t == 0:
                xt = holistic_feature
            else:
                it = seq[:, t - 1]
                xt = self.emb(it)

            for i in range(self.num_layers):
                if i == 0:
                    inp = xt
                else:
                    inp = h[i - 1, 0]
                h[i, 0], h[i, 1] = self.lstm_u[i](inp, (h[i, 0], h[i, 1]))
            ht = h[-1, 0]
            if t == 0:
                continue
            out_t, _ = self.att(ht, x_flatten, x_flatten)
            out_t = torch.cat((out_t, ht), dim=1)
            score = torch.softmax(self.cls(out_t), dim=1)
            score, idx = torch.max(score, dim=1)
            seq[:, t] = idx
            seq_score[:, t] = score
            end = end & (idx != self.START_TOKEN)
            if torch.sum(end) == 0:
                break

        # return seq, seq_score
        # words, word_scores = self.to_words(seq[:, 1:], seq_score[:, 1:]) # 2ms
        words, word_scores = vec2word(
            seq[:, 1:].cpu().numpy(),
            seq_score[:, 1:].cpu().numpy(),
            self.id2char, self.END_TOKEN)

        return words, word_scores


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.layer_norm(q)

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1,
                                                                        3)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1,
                                                                        3)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1,
                                                                        3)

        att = torch.matmul(q / self.scale, k.permute(0, 1, 3, 2))
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e10)
        att = torch.softmax(att, dim=-1)

        out = torch.matmul(self.dropout(att), v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, self.hidden_dim)

        out = self.dropout(self.fc_o(out))

        return out, att


class MultiHeadAttentionLayer_Fast(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()

        assert hidden_dim % n_heads == 0
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.att = nn.MultiheadAttention(hidden_dim, n_heads, dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.layer_norm(q).reshape(batch_size, 1, -1)

        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        out, att = self.att(q, k, v)

        out = out.reshape(batch_size, -1)

        return out, att
