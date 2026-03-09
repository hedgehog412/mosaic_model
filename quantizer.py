import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
  def __init__(self, num_embeddings = 512, embedding_dim = 64, commitment_cost = 0.25):
    super().__init__()

    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.commitment_cost = commitment_cost

    self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

  def forward(self, z_e):
    z = z_e.permute(0, 2, 3, 1).contiguous()
    z_flattened = z.view(-1, self.embedding_dim)

    z_sq = torch.sum(z_flattened ** 2, dim = 1, keepdim=True)
    e_sq = torch.sum(self.embedding.weight ** 2, dim = 1)
    ze = torch.matmul(z_flattened, self.embedding.weight.t())
    d = z_sq + e_sq.unsqueeze(0) - 2 * ze

    encoding_indices = torch.argmin(d, dim = 1)

    quantized = self.embedding(encoding_indices).view(z.shape)

    codebook_loss = F.mse_loss(quantized, z.detach())
    commitment_loss = F.mse_loss(z, quantized.detach())
    vq_loss = codebook_loss + self.commitment_cost * commitment_loss

    quantized_st = z + (quantized - z).detach()

    encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
    avg_probs = encodings.mean(0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

    z_q = quantized_st.permute(0, 3, 1, 2).contiguous()
    indices = encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])

    return z_q, vq_loss, perplexity, indices

class VectorQuantizerEMA(nn.Module):
  def __init__(self, decay = 0.99, epsilon = 1e-5, num_embeddings = 512, embedding_dim = 64, commitment_cost = 0.25):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.commitment_cost = commitment_cost

    self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
    self.register_buffer("ema_w", self.embedding.weight.data.clone())

    self.decay = decay
    self.epsilon = epsilon

  def forward(self, z_e):
    z = z_e.permute(0, 2, 3, 1).contiguous()
    z_flattened = z.view(-1, self.embedding_dim)

    z_sq = torch.sum(z_flattened ** 2, dim = 1, keepdim=True)
    e_sq = torch.sum(self.embedding.weight ** 2, dim = 1)
    ze = torch.matmul(z_flattened, self.embedding.weight.t())
    d = z_sq + e_sq.unsqueeze(0) - 2 * ze

    encoding_indices = torch.argmin(d, dim = 1)
    quantized = self.embedding(encoding_indices).view(z.shape)

    if self.training:
      enc_onehot = F.one_hot(encoding_indices, self.num_embeddings).type_as(z_flattened)
      self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * torch.sum(enc_onehot, 0)
      self.ema_w = self.decay * self.ema_w + (1 - self.decay) * torch.matmul(enc_onehot.t(), z_flattened)

      n = torch.sum(self.ema_cluster_size.data)
      cluster_size_smoothed = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
      new_weight = self.ema_w / cluster_size_smoothed.unsqueeze(1)
      self.embedding.weight.data.copy_(new_weight)

    commitment_loss = F.mse_loss(z, quantized.detach())
    vq_loss =  self.commitment_cost * commitment_loss

    quantized_st = z + (quantized - z).detach()

    encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
    avg_probs = encodings.mean(0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

    z_q = quantized_st.permute(0, 3, 1, 2).contiguous()
    indices = encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])

    return z_q, vq_loss, perplexity, indices