"""The attention cap must bring runaway heads exactly to the cap and leave the rest alone."""
import pytest
import torch

from araspellx.model.bert import BertForMaskedLM, make_config
from araspellx.train.stability import cap_attention, max_attention_scores


def _model(positions):
    torch.manual_seed(0)
    return BertForMaskedLM(make_config(vocab_size=50, layers=2, hidden=128, max_positions=64,
                                       position_embedding_type=positions)).train()


@pytest.mark.parametrize("positions", ["absolute", "relative_key"])
def test_cap_scales_only_runaway_heads_to_the_cap(positions):
    model = _model(positions)
    ids = torch.randint(5, 50, (4, 32))
    attention = model.bert.encoder.layer[1].attention.self
    with torch.no_grad():  # make head 1 of the last layer run away
        attention.query.weight[64:128] *= 300
        attention.query.bias[64:128] += 3
    before = max_attention_scores(model, ids)
    assert before[1, 1] > 50 and (before[0] < 50).all()

    capped = cap_attention(model, ids, cap=50.0)

    after = max_attention_scores(model, ids)
    assert [(layer, head) for layer, head, _ in capped] == [(1, 1)]
    assert torch.isclose(after[1, 1], torch.tensor(50.0), rtol=1e-4)
    others = torch.ones_like(after, dtype=torch.bool)
    others[1, 1] = False
    assert torch.allclose(after[others], before[others])
    assert model.training  # the probe does not leave the model in eval mode


def test_padding_is_ignored():
    model = _model("relative_key")
    ids = torch.randint(5, 50, (1, 20))
    padded = torch.cat([ids, torch.zeros(1, 12, dtype=torch.long)], 1)
    mask = torch.cat([torch.ones(1, 20, dtype=torch.long), torch.zeros(1, 12, dtype=torch.long)], 1)
    assert torch.allclose(max_attention_scores(model, padded, mask), max_attention_scores(model, ids), atol=1e-5)
