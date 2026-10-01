from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):

    def __init__(self, d_model: int, dropout: float=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, sequence: torch.Tensor, key_padding_mask: torch.Tensor | None=None) -> torch.Tensor:
        query = self.query.expand(sequence.shape[0], -1, -1)
        scores = torch.matmul(query, self.key(sequence).transpose(-2, -1))
        scores = scores / sequence.shape[-1] ** 0.5
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), -1000000000.0)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(weights, self.value(sequence)).squeeze(1)

class LDSPOMultimodalStudent(nn.Module):

    def __init__(self, numerical_dim: int=21, seq_len: int=30, d_model: int=128, n_heads: int=4, n_layers: int=3, dim_feedforward: int | None=None, dropout: float=0.1, num_actions: int=11, head_hidden: int=128, text_vocab_size: int=32768):
        super().__init__()
        self.numerical_dim = numerical_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.text_vocab_size = text_vocab_size
        feedforward_dim = dim_feedforward or 4 * d_model
        self.numerical_projection = nn.Sequential(nn.Linear(numerical_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.positional_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=feedforward_dim, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.numerical_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.numerical_pooling = AttentionPooling(d_model, dropout)
        self.token_embedding = nn.Embedding(text_vocab_size, d_model, padding_idx=0)
        self.article_encoder = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.news_pooling = AttentionPooling(d_model, dropout)
        self.no_news_embedding = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.numerical_fusion_projection = nn.Linear(d_model, d_model, bias=False)
        self.news_fusion_projection = nn.Linear(d_model, d_model, bias=False)
        self.fusion_gate = nn.Linear(2 * d_model, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.policy_head = nn.Sequential(nn.Linear(d_model, head_hidden), nn.LayerNorm(head_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden, num_actions))
        self.value_head = nn.Sequential(nn.Linear(d_model, head_hidden), nn.LayerNorm(head_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1))
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.xavier_uniform_(self.numerical_projection[0].weight)
        nn.init.zeros_(self.numerical_projection[0].bias)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head[-1].bias)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)

    def _encode_numerical(self, numerical: torch.Tensor) -> torch.Tensor:
        if numerical.ndim != 3:
            raise ValueError('numerical must have shape (batch, time, features)')
        if numerical.shape[-1] != self.numerical_dim:
            raise ValueError(f'expected {self.numerical_dim} numerical features, received {numerical.shape[-1]}')
        sequence_length = numerical.shape[1]
        states = self.numerical_projection(numerical)
        if sequence_length <= self.seq_len:
            positions = self.positional_embedding[:, :sequence_length]
        else:
            positions = F.interpolate(self.positional_embedding.transpose(1, 2), size=sequence_length, mode='linear', align_corners=False).transpose(1, 2)
        states = self.numerical_encoder(states + positions)
        return self.numerical_pooling(states)

    def _encode_news(self, news_token_ids: torch.Tensor, news_mask: torch.Tensor) -> torch.Tensor:
        if news_token_ids.ndim != 3:
            raise ValueError('news_token_ids must have shape (batch, articles, tokens)')
        article_mask = news_mask.bool()
        token_mask = news_token_ids.ne(0)
        embedded = self.token_embedding(news_token_ids.long())
        token_count = token_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        articles = (embedded * token_mask.unsqueeze(-1)).sum(dim=-2) / token_count.to(embedded.dtype)
        articles = self.article_encoder(articles)
        missing_news = ~article_mask.any(dim=1)
        if missing_news.any():
            articles = articles.clone()
            article_mask = article_mask.clone()
            articles[missing_news, 0] = self.no_news_embedding
            article_mask[missing_news, 0] = True
        return self.news_pooling(articles, key_padding_mask=~article_mask)

    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        numerical_state = self._encode_numerical(observations['numerical'])
        news_state = self._encode_news(observations['news_token_ids'], observations['news_mask'])
        gate = torch.sigmoid(self.fusion_gate(torch.cat([numerical_state, news_state], dim=-1)))
        fused_state = self.fusion_norm(self.numerical_fusion_projection(numerical_state) + gate * self.news_fusion_projection(news_state))
        return (self.policy_head(fused_state), self.value_head(fused_state))

    def get_action(self, observations: Dict[str, torch.Tensor], deterministic: bool=False, temperature: float=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        logits, value = self(observations)
        logits = torch.clamp(logits, -20.0, 20.0) / temperature
        distribution = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        return (action, distribution.log_prob(action), value.squeeze(-1), distribution.entropy())

    def evaluate_actions(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(observations)
        distribution = torch.distributions.Categorical(logits=torch.clamp(logits, -20.0, 20.0))
        return (distribution.log_prob(actions), value.squeeze(-1), distribution.entropy())
PolicyValueNetMultimodal = LDSPOMultimodalStudent
