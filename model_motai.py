import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Transformer_Based_Model


MODALITY_ALIASES = {
    'text': ('t',),
    'audio': ('a',),
    'visual': ('v',),
    'video': ('v',),
    'text_audio': ('t', 'a'),
    'audio_text': ('t', 'a'),
    'text_visual': ('t', 'v'),
    'text_video': ('t', 'v'),
    'visual_text': ('t', 'v'),
    'video_text': ('t', 'v'),
    'audio_visual': ('a', 'v'),
    'audio_video': ('a', 'v'),
    'visual_audio': ('a', 'v'),
    'video_audio': ('a', 'v'),
    't': ('t',),
    'a': ('a',),
    'v': ('v',),
    'ta': ('t', 'a'),
    'tv': ('t', 'v'),
    'av': ('a', 'v'),
}


def normalize_modalities(modalities):
    key = modalities.strip().lower().replace('+', '_').replace(',', '_').replace('-', '_')
    if key not in MODALITY_ALIASES:
        valid = ', '.join(sorted(MODALITY_ALIASES.keys()))
        raise ValueError('Unknown modalities: {}. Valid values: {}'.format(modalities, valid))
    return tuple(MODALITY_ALIASES[key])


def modality_name(modalities):
    mods = normalize_modalities(modalities) if isinstance(modalities, str) else tuple(modalities)
    names = {'t': 'text', 'a': 'audio', 'v': 'visual'}
    return '_'.join(names[m] for m in mods)


class ModalityMaskedGatedFusion(nn.Module):
    def __init__(self, hidden_size, active_modalities):
        super(ModalityMaskedGatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)
        active = torch.zeros(3, dtype=torch.bool)
        for idx, m in enumerate(('t', 'a', 'v')):
            if m in active_modalities:
                active[idx] = True
        self.register_buffer('active_mask', active.view(1, 1, 3, 1))

    def forward(self, text_rep, audio_rep, visual_rep):
        reps = torch.cat([
            text_rep.unsqueeze(-2),
            audio_rep.unsqueeze(-2),
            visual_rep.unsqueeze(-2),
        ], dim=-2)
        logits = torch.cat([
            self.fc(text_rep).unsqueeze(-2),
            self.fc(audio_rep).unsqueeze(-2),
            self.fc(visual_rep).unsqueeze(-2),
        ], dim=-2)
        logits = logits.masked_fill(~self.active_mask, -1e4)
        weights = self.softmax(logits)
        return torch.sum(weights * reps, dim=-2, keepdim=False)


class ModalityAblationModel(Transformer_Based_Model):
    def __init__(self, *args, modalities='text_audio', **kwargs):
        self.active_modalities = normalize_modalities(modalities)
        super(ModalityAblationModel, self).__init__(*args, **kwargs)
        self.last_gate = ModalityMaskedGatedFusion(args[7] if len(args) > 7 else kwargs['hidden_dim'],
                                                   self.active_modalities)

    def _active(self, modality):
        return modality in self.active_modalities

    def _branch(self, source, target, encoder, gate, source_active, target_active, u_mask, spk_embeddings):
        if not target_active or not source_active:
            return torch.zeros_like(target)
        return gate(encoder(source, target, u_mask, spk_embeddings))

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, llm_features=None, llm_mask=None):
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2 * torch.ones(origin_spk_idx[i].size(0) - x)).int().to(spk_idx.device)
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9 * torch.ones(origin_spk_idx[i].size(0) - x)).int().to(spk_idx.device)
        spk_embeddings = self.speaker_embeddings(spk_idx)

        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)

        t_on = self._active('t')
        a_on = self._active('a')
        v_on = self._active('v')

        t_t = self._branch(textf, textf, self.t_t, self.t_t_gate, t_on, t_on, u_mask, spk_embeddings)
        a_t = self._branch(acouf, textf, self.a_t, self.a_t_gate, a_on, t_on, u_mask, spk_embeddings)
        v_t = self._branch(visuf, textf, self.v_t, self.v_t_gate, v_on, t_on, u_mask, spk_embeddings)

        a_a = self._branch(acouf, acouf, self.a_a, self.a_a_gate, a_on, a_on, u_mask, spk_embeddings)
        t_a = self._branch(textf, acouf, self.t_a, self.t_a_gate, t_on, a_on, u_mask, spk_embeddings)
        v_a = self._branch(visuf, acouf, self.v_a, self.v_a_gate, v_on, a_on, u_mask, spk_embeddings)

        v_v = self._branch(visuf, visuf, self.v_v, self.v_v_gate, v_on, v_on, u_mask, spk_embeddings)
        t_v = self._branch(textf, visuf, self.t_v, self.t_v_gate, t_on, v_on, u_mask, spk_embeddings)
        a_v = self._branch(acouf, visuf, self.a_v, self.a_v_gate, a_on, v_on, u_mask, spk_embeddings)

        t_out = self.features_reduce_t(torch.cat([t_t, a_t, v_t], dim=-1)) if t_on else torch.zeros_like(textf)
        a_out = self.features_reduce_a(torch.cat([a_a, t_a, v_a], dim=-1)) if a_on else torch.zeros_like(acouf)
        v_out = self.features_reduce_v(torch.cat([v_v, t_v, a_v], dim=-1)) if v_on else torch.zeros_like(visuf)

        all_out = self.last_gate(t_out, a_out, v_out)
        self.last_reliability_weights = None
        self.last_contrast_features = all_out

        t_final = self.t_output_layer(t_out)
        a_final = self.a_output_layer(a_out)
        v_final = self.v_output_layer(v_out)
        all_final = self.all_output_layer(all_out)

        t_log_prob = F.log_softmax(t_final, 2)
        a_log_prob = F.log_softmax(a_final, 2)
        v_log_prob = F.log_softmax(v_final, 2)
        all_log_prob = F.log_softmax(all_final, 2)
        all_prob = F.softmax(all_final, 2)

        kl_t_log_prob = F.log_softmax(t_final / self.temp, 2)
        kl_a_log_prob = F.log_softmax(a_final / self.temp, 2)
        kl_v_log_prob = F.log_softmax(v_final / self.temp, 2)
        kl_all_prob = F.softmax(all_final / self.temp, 2)

        return t_log_prob, a_log_prob, v_log_prob, all_log_prob, all_prob, \
            kl_t_log_prob, kl_a_log_prob, kl_v_log_prob, kl_all_prob
