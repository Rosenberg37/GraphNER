import torch
from torch import nn as nn, Tensor


class Graph(nn.Module):
    def __init__(self, num_iters: int, layer_kargs: dict):
        super().__init__()
        self.layers = nn.ModuleList([UpdateLayer(**layer_kargs) for _ in range(num_iters)])

    def forward(self, context: Tensor, types: Tensor, context_mask: Tensor) -> tuple[Tensor, Tensor]:
        """

        :param context: [batch_size, sentence_length, hidden_size]
        :param types: [batch_size, types_num, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return:
            context: [batch_size, sentence_length, hidden_size]
        """
        for layer in self.layers:
            types, context = layer(types, context, context_mask)
        return types, context

    def get_attentions(self, context: Tensor, types: Tensor, context_mask: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        context_attentions, types_attentions = list(), list()
        for layer in self.layers:
            context, types, context_attention, types_attention = layer.attn_forward(types, context, context_mask)
            context_attentions.append(context_attention)
            types_attentions.append(types_attention)
        return context_attentions, types_attentions


class UpdateLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, window_size: int, use_gate: bool, use_hybrid: bool, updates: list[str]):
        super(UpdateLayer, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.updates = updates

        edge_block = EdgeBlock(hidden_size, num_heads, use_hybrid)

        self.norms = nn.ModuleDict()
        if 'types' in updates:
            self.types_block = TypesBlock(hidden_size, num_heads, edge_block, use_gate)
            self.norms['types'] = nn.LayerNorm(hidden_size)

        if 'context' in updates:
            self.context_block = ContextBlock(hidden_size, num_heads, window_size, edge_block, use_gate)
            self.norms['context'] = nn.LayerNorm(hidden_size)

    def forward(self, types: Tensor, context: Tensor, context_mask: Tensor):
        """

        :param types: [batch_size, types_num, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return:
            types: [batch_size, types_num, hidden_size]
            context: [batch_size, sentence_length, hidden_size]
        """
        if 'context' in self.updates:
            context = self.norms['context'](context + self.context_block(context, context_mask, types)[0])
        if 'types' in self.updates:
            types = self.norms['types'](types + self.types_block(types, context, context_mask)[0])
        return types, context

    def attn_forward(self, types: Tensor, context: Tensor, context_mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        :param types: [batch_size, types_num, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return:
            context_attention: [batch_size, num_heads, types_num + sentence_length, types_num + sentence_length]
            types_attention: [batch_size, num_heads, types_num + sentence_length, types_num + sentence_length]
        """
        device = types.device
        batch_size, types_num = types.shape[:2]
        batch_size, sentence_length = context.shape[:2]

        context_update, types_attention, window_attention = self.context_block(context, context_mask, types)
        attentions_list = list()
        padding = torch.zeros([batch_size, self.num_heads, sentence_length], device=device)
        for i, attn in enumerate(window_attention.permute(2, 0, 1, 3)):  # [batch_size, num_heads, value_size]
            attn_left = max(self.window_size - i, 0)
            attn_right = 2 * self.window_size + 2 - max(i + self.window_size - sentence_length + 2, 0)
            attn = attn[..., attn_left:attn_right]

            padding_before = padding[..., :max(i - self.window_size, 0)]
            padding_last = padding[..., min(i + self.window_size + 1, sentence_length):]
            attentions_list.append(torch.cat([padding_before, attn, padding_last], dim=-1))
        context_attention = torch.stack(attentions_list, dim=-2)
        context_attention = torch.cat([types_attention, context_attention], dim=-1)
        padding = torch.zeros([batch_size, self.num_heads, types_num, types_num + sentence_length], device=device)
        context_attention = torch.cat([padding, context_attention], dim=-2)

        types_update, types_attention = self.types_block(types, context, context_mask)
        padding = torch.zeros([batch_size, self.num_heads, types_num, types_num], device=device)
        types_attention = torch.cat([padding, types_attention], dim=-1)
        padding = torch.zeros([batch_size, self.num_heads, sentence_length, types_num + sentence_length], device=device)
        types_attention = torch.cat([types_attention, padding], dim=-2)

        context = self.norms['context'](context + context_update)
        types = self.norms['types'](types + types_update)

        return context, types, context_attention, types_attention


class TypesBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, edge_block: nn.Module, use_gate: bool):
        super(TypesBlock, self).__init__()
        head_dim = hidden_size // num_heads
        self.use_gate = use_gate

        self.transforms = nn.ModuleDict({
            'types': nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Unflatten(-1, [num_heads, head_dim])
            ),
            'context': nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Unflatten(-1, [num_heads, head_dim])
            )
        })
        self.edge_block = edge_block

        if use_gate:
            self.cell = GateCell(hidden_size)
        else:
            self.out = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh()
            )

    def forward(self, types: Tensor, context: Tensor, context_mask: Tensor) -> tuple[Tensor, Tensor]:
        """

        :param types: [batch_size, types_num, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return:
            types: [batch_size, types_num, hidden_size]
            weights: [batch_size, types_num, sentence_length]
        """
        types_h = self.transforms['types'](types)
        context_h = self.transforms['context'](context)
        score = self.edge_block.cross_attn(types_h, context_h)
        score = score.masked_fill(~context_mask.unsqueeze(1).unsqueeze(1), -1e12)
        weights = torch.softmax(score, dim=-1)
        update = torch.einsum('bntl,blnd->btnd', weights, context_h).flatten(-2)

        if self.use_gate:
            update = self.cell(update + types, types)
        else:
            update = self.out(update + types)
        return update, weights


class ContextBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, window_size: int, edge_block: nn.Module, use_gate: bool):
        super(ContextBlock, self).__init__()
        head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.use_gate = use_gate

        self.transforms = nn.ModuleDict({
            'types': nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Unflatten(-1, [num_heads, head_dim])
            ),
            'context': nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Unflatten(-1, [num_heads, head_dim])
            )
        })
        self.edge_block = edge_block

        if use_gate:
            self.cell = GateCell(hidden_size)
        else:
            self.out = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh()
            )

    def forward(self, context: Tensor, context_mask: Tensor, types: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """

        :param types: [batch_size, hidden_size]
        :param context: [batch_size, sentence_length, hidden_size]
        :param context_mask: [batch_size, sentence_length]
        :return:
            context: [batch_size, sentence_length, hidden_size]
            weights: [batch_size, sentence_length, types_num + window_size * 2 + 1]
        """
        types_num = types.shape[1]

        types_h = self.transforms['types'](types)
        context_h = self.transforms['context'](context)

        types_s = self.edge_block.cross_attn(context_h, types_h)
        context_s, value_h = self.edge_block.self_attn(context_h, context_mask, self.window_size)

        weights = torch.softmax(torch.cat([types_s, context_s], dim=-1), dim=-1)
        types_weights, context_weights = weights.split([types_num, self.window_size * 2 + 1], -1)

        update_types = torch.einsum('bnlt,btnd->blnd', types_weights, types_h)
        update_context = torch.einsum('bnlv,blvnd->blnd', context_weights, value_h)
        update = (update_types + update_context).flatten(-2)

        if self.use_gate:
            update = self.cell(update + context, context)
        else:
            update = self.out(update + context)
        return update, types_weights, context_weights


class EdgeBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, use_hybrid: bool, negative_slope: float = 5):
        super(EdgeBlock, self).__init__()
        self.use_hybrid = use_hybrid
        head_dim = hidden_size // num_heads
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        gain = nn.init.calculate_gain('leaky_relu', negative_slope)

        self.upon = nn.Parameter(torch.empty(num_heads, head_dim))
        nn.init.xavier_uniform_(self.upon, gain)
        self.down = nn.Parameter(torch.empty(num_heads, head_dim))
        nn.init.xavier_uniform_(self.down, gain)
        if use_hybrid:
            self.cross = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
            nn.init.xavier_uniform_(self.cross, gain)

    def cross_attn(self, query_h: Tensor, value_h: Tensor) -> Tensor:
        """

        :param query_h: [batch_size, query_size, num_heads, head_dim]
        :param value_h: [batch_size, value_size, num_heads, head_dim]
        :return: [batch_size, query_size, num_heads, head_dim]
        """
        upon_s = torch.einsum('bqnd,nd->bnq', query_h, self.upon)
        down_s = torch.einsum('bvnd,nd->bnv', value_h, self.down)
        concat_score = upon_s.unsqueeze(-1) + down_s.unsqueeze(-2)
        if self.use_hybrid:
            product_score = torch.einsum('bqnd,ndh,bvnh->bnqv', query_h, self.cross, value_h)
            return self.leaky_relu(concat_score + product_score)
        else:
            return self.leaky_relu(concat_score)

    def self_attn(self, hidden: Tensor, mask: Tensor, window_size: int):
        """

        :param window_size: the size of self connect window
        :param hidden: [batch_size, sentence_length, num_heads, head_dim]
        :param mask: [batch_size, sentence_length]
        :return: [batch_size, sentence_length, num_heads, head_dim]
        """
        length = mask.shape[1]

        indices_u = torch.cat([torch.zeros(window_size + 1, dtype=torch.long), torch.arange(1, window_size + 1)])
        indices_u = (torch.arange(length).unsqueeze(1) + indices_u).to(hidden.device)

        indices_d = torch.cat([torch.arange(-window_size, 0), torch.zeros(window_size + 1, dtype=torch.long)])
        indices_d = (torch.arange(length).unsqueeze(1) + indices_d).to(hidden.device)

        indices_mask = indices_d.ge(0) & indices_u.lt(length)
        indices_u, indices_d = indices_u % length, indices_d % length

        upon_s = torch.einsum('blnd,nd->bnl', hidden, self.upon)
        down_s = torch.einsum('blnd,nd->bnl', hidden, self.down)
        concat_score = upon_s[:, :, indices_u] + down_s[:, :, indices_d]

        window_indices = torch.cat([indices_d[:, :0], indices_u[:, 0:]], dim=-1)
        mask = mask[:, window_indices % length] & indices_mask

        value_h = hidden[:, window_indices]

        if self.use_hybrid:
            product_score = torch.einsum('blnd,ndh,blvnh->bnlv', hidden, self.cross, value_h)
            window_score = self.leaky_relu(concat_score + product_score).masked_fill(~mask.unsqueeze(1), -1e12)
        else:
            window_score = self.leaky_relu(concat_score).masked_fill(~mask.unsqueeze(1), -1e12)

        return window_score, value_h


class GateCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim

        self.matrix_xr = nn.Linear(input_dim, hidden_dim)
        self.matrix_hr = nn.Linear(input_dim, hidden_dim)

        self.matrix_xz = nn.Linear(input_dim, hidden_dim)
        self.matrix_hz = nn.Linear(input_dim, hidden_dim)

        self.matrix_xn = nn.Linear(input_dim, hidden_dim)
        self.matrix_hn = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tensor:
        """

        :param x: [*, input_dim]
        :param h: [*, input_dim]
        :return: [*, hidden_dim]
        """
        r = torch.sigmoid(self.matrix_xr(x) + self.matrix_hr(h))
        z = torch.sigmoid(self.matrix_xz(x) + self.matrix_hz(h))
        n = torch.tanh(self.matrix_xn(x) + r * self.matrix_hn(h))
        return (1 - z) * n + z * h
