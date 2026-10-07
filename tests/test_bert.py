"""Our BERT must be numerically identical to Hugging Face's classes."""
import torch
import transformers

from araspellx.model.bert import BertForMaskedLM, BertForTokenClassification, make_config

ATOL = 1e-5


def _config(**kwargs):
    return make_config(vocab_size=120, num_labels=7, layers=2, hidden=128,
                       max_positions=64, attn_implementation="eager", **kwargs)


def _batch():
    torch.manual_seed(0)
    ids = torch.randint(5, 120, (3, 40))
    mask = torch.ones_like(ids)
    mask[1, 25:] = 0  # padded rows
    mask[2, 10:] = 0
    ids[mask == 0] = 0
    return ids, mask


def test_token_classification_matches_hugging_face():
    torch.manual_seed(1)
    hf = transformers.BertForTokenClassification(_config()).eval()
    ours = BertForTokenClassification(_config()).eval()
    ours.load_hf_state_dict(hf.state_dict())
    ids, mask = _batch()
    with torch.no_grad():
        expected = hf(input_ids=ids, attention_mask=mask).logits
        actual = ours(ids, mask)
    real = mask.bool()
    assert torch.allclose(actual[real], expected[real], atol=ATOL)


def test_masked_lm_matches_hugging_face():
    torch.manual_seed(2)
    hf = transformers.BertForMaskedLM(_config()).eval()
    ours = BertForMaskedLM(_config()).eval()
    ours.load_hf_state_dict(hf.state_dict())
    ids, mask = _batch()
    with torch.no_grad():
        expected = hf(input_ids=ids, attention_mask=mask).logits
        actual = ours(ids, mask)
    real = mask.bool()
    assert torch.allclose(actual[real], expected[real], atol=ATOL)
    assert ours.cls.predictions.decoder.weight is ours.bert.embeddings.word_embeddings.weight


def test_our_checkpoints_load_in_hugging_face(tmp_path):
    torch.manual_seed(3)
    for ours, hf_class in [
        (BertForTokenClassification(_config()), transformers.BertForTokenClassification),
        (BertForMaskedLM(_config()), transformers.BertForMaskedLM),
    ]:
        ours.eval().save_pretrained(tmp_path / ours.architecture)
        hf = hf_class.from_pretrained(tmp_path / ours.architecture, attn_implementation="eager").eval()
        ids, mask = _batch()
        with torch.no_grad():
            real = mask.bool()
            assert torch.allclose(ours(ids, mask)[real], hf(input_ids=ids, attention_mask=mask).logits[real], atol=ATOL)
        reloaded = type(ours).from_pretrained(tmp_path / ours.architecture).eval()
        with torch.no_grad():
            assert torch.equal(ours(ids, mask), reloaded(ids, mask))


def test_training_step_runs():
    torch.manual_seed(4)
    model = BertForTokenClassification(_config()).train()
    ids, mask = _batch()
    labels = torch.randint(0, 7, ids.shape)
    logits = model(ids, mask)
    loss = torch.nn.functional.cross_entropy(logits[mask.bool()], labels[mask.bool()])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
