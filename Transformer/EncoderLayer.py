class EncoderLayer(nn.Module):
	    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    // 하나의 임베딩이 복제되어 Query, Key, Value로 입력되는 방식
    def forward(self, src, src_mask):

        // src: [batch_size, src_len, hidden_dim]
        // src_mask: [batch_size, src_len]

        // self attention
        //  필요한 경우 마스크(mask) 행렬을 이용하여 어텐션(attention)할 단어를 조절 가능
        _src, _ = self.self_attention(src, src, src, src_mask)

        // dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        // src: [batch_size, src_len, hidden_dim]

        // position-wise feedforward
        _src = self.positionwise_feedforward(src)

        // dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        // src: [batch_size, src_len, hidden_dim]

        return src
