# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from . import transformer as tr

from .models import META_ARCHITECTURES as registry
from .feature_head import build_feature_head

class LSTR(nn.Module):

    def __init__(self, cfg):
        super(LSTR, self).__init__()

        # Build long feature heads
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.long_enabled = self.long_memory_num_samples > 0
        if self.long_enabled:
            self.feature_head_long = build_feature_head(cfg)

        # Build work feature head
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.work_enabled = self.work_memory_num_samples > 0
        if self.work_enabled:
            self.feature_head_work = build_feature_head(cfg)


        self.d_model = self.feature_head_work.d_model
        self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.activation = cfg.MODEL.LSTR.ACTIVATION
        self.num_classes = cfg.DATA.NUM_CLASSES

        # Build position encoding
        self.pos_encoding = tr.PositionalEncoding(self.d_model, self.dropout)

        # Build LSTR encoder1
        if self.long_enabled:
            self.enc1_queries = nn.ModuleList()
            self.enc1_modules = nn.ModuleList()

            self.enc1_queries.append(nn.Embedding(32, self.d_model))
            enc1_layer = tr.TransformerSparseDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation,370)
            self.enc1_modules.append(tr.TransformerDecoder(
                enc1_layer, 1, tr.layer_norm(self.d_model, True)))

        # Build LSTR encoder2
        if self.long_enabled:
            self.enc2_queries = nn.ModuleList()
            self.enc2_modules = nn.ModuleList()

            self.enc2_queries.append(nn.Embedding(16, self.d_model))
            enc2_layer = tr.TransformerSparseDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation,370)
            self.enc2_modules.append(tr.TransformerDecoder(
                enc2_layer, 1, tr.layer_norm(self.d_model, True)))

            self.enc2_queries.append(nn.Embedding(32, self.d_model))
            enc2_layer = tr.TransformerSparseDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation,370)
            self.enc2_modules.append(tr.TransformerDecoder(
                enc2_layer, 2, tr.layer_norm(self.d_model, True)))


        # Build LSTR decoder1
        dec_layer1 = tr.TransformerSparseDecoderLayer(
            self.d_model, self.num_heads, self.dim_feedforward,
            self.dropout, self.activation,370)
        self.dec_modules1 = tr.TransformerDecoder(
            dec_layer1, 1, tr.layer_norm(self.d_model, True))


        # Build classifier
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None):
        if self.long_enabled:
            # Compute long memories
            long_memories = self.pos_encoding(self.feature_head_long(
                visual_inputs[:, :self.long_memory_num_samples],
                motion_inputs[:, :self.long_memory_num_samples],
            ).transpose(0,1))

            long_memory1 = long_memories
            long_memory2 = long_memories
            long_memory3 = long_memories

            #encoder1
            if len(self.enc1_modules) > 0:
                enc1_queries = [
                    enc1_query.weight.unsqueeze(1).repeat(1, long_memory1.shape[1], 1)
                    if enc1_query is not None else None
                    for enc1_query in self.enc1_queries
                ]
                if enc1_queries[0] is not None:
                    long_memory1 = self.enc1_modules[0](enc1_queries[0], long_memory1,
                                                        memory_key_padding_mask=memory_key_padding_mask)

            #encoder2
            if len(self.enc2_modules) > 0:
                enc2_queries = [
                    enc2_query.weight.unsqueeze(1).repeat(1, long_memory2.shape[1], 1)
                    if enc2_query is not None else None
                    for enc2_query in self.enc2_queries
                ]
                if enc2_queries[0] is not None:
                    long_memory2 = self.enc2_modules[0](enc2_queries[0], long_memory2,
                                                        memory_key_padding_mask=memory_key_padding_mask)

                if enc2_queries[1] is not None:
                    long_memory2 = self.enc2_modules[1](long_memory1, long_memory2)



        # Concatenate memories
        enc1_score = self.classifier(long_memory1)
        enc2_score = self.classifier(long_memory2)

        if self.work_enabled:
            # Compute work memories
            work_memories = self.pos_encoding(self.feature_head_work(
                visual_inputs[:, self.long_memory_num_samples:],
                motion_inputs[:, self.long_memory_num_samples:],
            ).transpose(0, 1), padding=self.long_memory_num_samples)

            # Build mask
            mask = tr.generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)

            # Compute output
            output = self.dec_modules1(
                work_memories,
                memory=long_memory1,
                tgt_mask=mask,
            )
            output = self.dec_modules1(
                output,
                memory=long_memory2,
                tgt_mask=mask,
            )

        # Compute classification score
        score = self.classifier(output)

        return score.transpose(0, 1),enc1_score.transpose(0,1), enc2_score.transpose(0,1)


@registry.register('LSTR')
class LSTRStream(LSTR):

    def __init__(self, cfg):
        super(LSTRStream, self).__init__(cfg)

        ############################
        # Cache for stream inference
        ############################
        self.long_memories_cache = None
        self.compressed_long_memories_cache = None

    def stream_inference(self,
                         long_visual_inputs,
                         long_motion_inputs,
                         work_visual_inputs,
                         work_motion_inputs,
                         memory_key_padding_mask=None):
        assert self.long_enabled, 'Long-term memory cannot be empty for stream inference'
        assert len(self.enc_modules) > 0, 'LSTR encoder cannot be disabled for stream inference'

        if (long_visual_inputs is not None) and (long_motion_inputs is not None):
            # Compute long memories
            long_memories = self.feature_head_long(
                long_visual_inputs,
                long_motion_inputs,
            ).transpose(0, 1)

            if self.long_memories_cache is None:
                self.long_memories_cache = long_memories
            else:
                self.long_memories_cache = torch.cat((
                    self.long_memories_cache[1:], long_memories
                ))

            long_memories = self.long_memories_cache
            pos = self.pos_encoding.pe[:self.long_memory_num_samples, :]

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in self.enc_queries
            ]
            # Encode long memories
            long_memories = self.enc_modules[0].stream_inference(enc_queries[0], long_memories, pos,
                                                                 memory_key_padding_mask=memory_key_padding_mask)
            self.compressed_long_memories_cache  = long_memories
            for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)
        else:
            long_memories = self.compressed_long_memories_cache

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in self.enc_queries
            ]

            # Encode long memories
            for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            work_memories = self.pos_encoding(self.feature_head_work(
                work_visual_inputs,
                work_motion_inputs,
            ).transpose(0, 1), padding=self.long_memory_num_samples)

            # Build mask
            mask = tr.generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)

            # Compute output
            if self.long_enabled:
                output = self.dec_modules(
                    work_memories,
                    memory=memory,
                    tgt_mask=mask,
                )
            else:
                output = self.dec_modules(
                    work_memories,
                    src_mask=mask,
                )

        # Compute classification score
        score = self.classifier(output)

        return score.transpose(0, 1)
