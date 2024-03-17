import torch
from torch import nn, Tensor
from typing import Dict, Tuple
from typing import Iterable, Optional, List
import torch.nn.functional as F
import coremltools as ct

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head = 6

        n_state = 384
        n_ctx = 448

        self.embedding = nn.Embedding(10000, n_state)
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        self.mask = torch.empty(n_ctx, n_ctx).fill_(-torch.inf).triu_(1)

        self.n_state = n_state

    def forward(
        self,
        x: Tensor,
        seq_len: Optional[Tensor] = None,
        key_cache: Optional[Tensor] = None,
        value_cache: Optional[Tensor] = None
    ):
        x = self.embedding(x)
        q = self.query(x)

        if key_cache is not None:
            new_k_col = self.key(x)
            new_v_col = self.value(x)

            # Calculate kv_cache for the given x.
            # To generate a model with a fixed shape, we need to add a padding in `seq` dimension to make CoreML conversion don't crash
            # logically equals to `torch.cat([key_cache, new_k_col])` but with key_cache trim to given seq_len and have padding removed
            k = torch.cat([key_cache[:, :seq_len], new_k_col], dim=1)[:, 1:]
            v = torch.cat([value_cache[:, :seq_len], new_v_col], dim=1)[:, 1:]

            wv, qk = self.qkv_attention(q, k, v, self.mask)
            return self.out(wv), new_k_col, new_v_col
        
        else:
            # no cache

            k = self.key(x)
            v = self.value(x)

            wv, qk = self.qkv_attention(q, k, v, self.mask)
            return self.out(wv), k, v
    
    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
       
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

def run_with_cache(model: SelfAttention, x: List[int]):
    key_cache = torch.zeros([1, 1, model.n_state], dtype=torch.float)
    value_cache = torch.zeros([1, 1, model.n_state], dtype=torch.float)
    for idx, sliced_x in enumerate(x):
        input = torch.full((1,1), sliced_x).to(torch.int64)
        seq_len = torch.full((1, 1), idx + 1).to(torch.int64)
        y, k_col, v_col = model(input, seq_len, key_cache, value_cache)
        key_cache = torch.cat([key_cache, k_col], dim=1)
        value_cache = torch.cat([value_cache, v_col], dim=1)
        if idx == len(x) - 1:
            return y, key_cache[:, 1:], value_cache[:, 1:]

    return None

def run_without_cache(model: SelfAttention, x: List[int]):
    x = torch.Tensor(x).unsqueeze(0)
    y, key, value = model(x.to(torch.int64))
    return y[:, -1:], key, value

def trace_with_cache(model, file_path):
    max_seq_len = 128

    x = torch.clamp(torch.randn((1, 1)).to(torch.int64), max=10000, min=0)
    seq_len = torch.full((1,1), 1).to(torch.int64)
    key_cache = torch.zeros([1, max_seq_len, model.n_state], dtype=torch.float)
    value_cache = torch.zeros([1, max_seq_len, model.n_state], dtype=torch.float)
    traced = torch.jit.trace(model.eval(), example_inputs=[x, seq_len, key_cache, value_cache])

    inputs = [
        ct.TensorType(name="x", shape=x.shape),
        ct.TensorType(name="seq_len", shape=seq_len.shape),
        ct.TensorType(name="key_cache", shape=key_cache.shape),
        ct.TensorType(name="value_cache", shape=value_cache.shape)
    ]
    outputs = [
        ct.TensorType(name="y"),
        ct.TensorType(name="new_k_col"),
        ct.TensorType(name="new_v_col")
    ]
    ml = ct.convert(traced, inputs=inputs, outputs=outputs, minimum_deployment_target=ct.target.iOS16, source="pytorch", convert_to="mlprogram")
    ml.save(file_path)

def trace_without_cache(model, file_path):
    seq_len = 128
    x = torch.clamp(torch.randn((1, seq_len)).to(torch.int64), max=10000, min=0)
    traced = torch.jit.trace(model.eval(), example_inputs=[x])
    shape = (1, seq_len)

    inputs = [ct.TensorType(name="x", shape=shape)]
    outputs = [
        ct.TensorType(name="y"),
        ct.TensorType(name="key_cache"),
        ct.TensorType(name="value_cache")
    ]
    ml = ct.convert(traced, inputs=inputs, outputs=outputs, minimum_deployment_target=ct.target.iOS16, source="pytorch", convert_to="mlprogram")
    ml.save(file_path)


if __name__ == '__main__':
    model = SelfAttention()
    
    x = [215,684,233,2235,8795,9687,6547,3434,215,684,233,2235,8795,9687,6547,3434]
    lhs, lKey, lValue = run_with_cache(model=model, x=x)
    rhs, rKey, rValue = run_without_cache(model=model, x=x)
    assert torch.allclose(lKey, rKey, atol=0.005)
    assert torch.allclose(lValue, rValue, atol=0.005)
    assert torch.allclose(lhs, rhs, atol=0.005)

    trace_with_cache(model, "./self_attention_with_cache")
    trace_without_cache(model, "./self_attention_without_cache")