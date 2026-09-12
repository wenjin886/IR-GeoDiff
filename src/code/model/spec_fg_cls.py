import torch
from torch import nn, optim
import torch.nn.functional as F

import lightning as L
from torchmetrics.classification import MultilabelAccuracy
import copy
from .spec_utils import (
    Embeddings,
    PositionalEncoding,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    Encoder, EncoderLayer,
    Decoder, DecoderLayer,
    EmbedPatchAttention
)




class SpecFuncGroupsClsModel(L.LightningModule):
    def __init__(self, 
                 formula_vocab_size=26, num_fg_cls=17,
                 spec_len=3200, patch_len=64, spec_vocab=100,
                 n_spec_attention_head=8,
                 n_spec_f_encoder_head=8,
                 n_spec_f_encoder_layer=4,
                 n_fg_decoder_head=8,
                 n_fg_decoder_layer=4,
                 fg_decoder_dropout=0.1,
                 device="cuda",
                 d_model=512,
                 cls_hidden_dim=256,
                 lr=1,
                 warm_up_step=4000,
                 use_formula=True,
                 ):
        super().__init__()
        self.device_ = device
        self.d_model = d_model
        self.spec_patch_num = int(spec_len/patch_len)
        self.use_formula = use_formula
        
        c = copy.deepcopy
        pe = PositionalEncoding(d_model, dropout=0.1)
        spec_patch_att = EmbedPatchAttention(spec_len=spec_len, 
                                              patch_len=patch_len, 
                                              src_vocab=spec_vocab,
                                              d_model=d_model,
                                              h=n_spec_attention_head)
        
        # Encoder part
        self.spec_embed = nn.Sequential(spec_patch_att, c(pe))
        if self.use_formula:
            self.formula_embed = nn.Sequential(Embeddings(d_model=d_model, vocab=formula_vocab_size), c(pe))
        
        encoder_attn = MultiHeadedAttention(h=n_spec_f_encoder_head,d_model=d_model)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=2048, dropout=0.1)
        self.spec_atom_encoder = Encoder(EncoderLayer(d_model, c(encoder_attn),c(ff),0.1), 
                                         N=n_spec_f_encoder_layer)


        # Decoder part
        self.fg_queries = nn.Parameter(torch.randn(num_fg_cls, d_model))
        
        decoder_att = MultiHeadedAttention(h=n_fg_decoder_head, d_model=d_model,
                                           dropout=fg_decoder_dropout)
        self.decoder = Decoder(DecoderLayer(size=d_model, self_attn=c(decoder_att), src_attn=c(encoder_attn), feed_forward=c(ff), dropout=0.1),
                                      N=n_fg_decoder_layer)

        # Classification part
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, cls_hidden_dim),
            # nn.ReLU(),
            nn.SiLU(),
            nn.Linear(cls_hidden_dim, 1)
        )     

        self.criterion = nn.BCEWithLogitsLoss()
        self.acc_per_fg = MultilabelAccuracy(num_labels=num_fg_cls, average="none")
        self.lr = lr
        self.warm_up_step = warm_up_step
        
        self.save_hyperparameters()
        self.__init_weights__()
    
    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
    
    def encode(self, formula_ids=None, spectra=None):
        assert spectra is not None, f"Please check the input of spectra"
        spectra_ = self.spec_embed(spectra)
        if self.use_formula:
            formula_ = self.formula_embed(formula_ids)
            src_ = torch.cat((spectra_, formula_), dim=1)
            src_mask = self._get_src_mask(formula_ids)
        else:
            src_ = spectra_
            src_mask = torch.ones(spectra.shape[0], self.spec_patch_num, dtype=torch.bool).unsqueeze(-2).to(self.device_) # (B, 1, S)

        return self.spec_atom_encoder(src_, src_mask), src_mask
    
    def decode(self, fg_queries, spec_formula_encout, src_mask):
        bs = spec_formula_encout.shape[0]
        tgt_mask = torch.ones(bs, 1, fg_queries.shape[0], dtype=torch.bool).to(self.device_)
        fg_queries_ = fg_queries.unsqueeze(0).repeat(bs, 1, 1)
        # print(fg_queries_.shape, spec_formula_encout.shape, src_mask.shape, tgt_mask.shape)
        fg_queries_ =  self.decoder(fg_queries_,
                            memory=spec_formula_encout,
                            src_mask=src_mask, 
                            tgt_mask=tgt_mask
                            )
        return fg_queries_
        
        
    def forward(self, spectra, formula_ids=None):

        spec_formula_encout, src_mask = self.encode(formula_ids, spectra)
        fg_queries_ = self.decode(self.fg_queries, spec_formula_encout, src_mask)
        logits = self.cls_head(fg_queries_).squeeze(-1)
        return logits, fg_queries_
  
    def _get_src_mask(self, formula_ids):
        """
        No padding mask for spec, only for atoms' token.
        """
        (batch_size, _) =  formula_ids.shape

        spectra_att_mask = torch.ones(batch_size, self.spec_patch_num, dtype=torch.bool).to(self.device_) # (B, S)
        formula_att_mask = (formula_ids != 0).to(self.device_) 
        src_mask = torch.cat((spectra_att_mask, formula_att_mask), dim=-1).unsqueeze(-2)  # (B, S+N)

        return src_mask
    
    def _cal_loss(self, logits, fg_onehot):
        loss = self.criterion(logits, fg_onehot)
        return loss

    def _cal_acc(self, logits, fg_onehot):
        acc = self.acc_per_fg(logits.sigmoid(), fg_onehot)
        return acc
    
    def training_step(self, batch, batch_idx):
        spectra = batch["spectra"].to(self.device_)
        formula_ids = batch["formula"].to(self.device_).int()
        fg_onehot = batch["func_groups"].to(self.device_, torch.float)
        
        bs = spectra.shape[0]
        
        logits, _ =  self(spectra, formula_ids)
        
        loss = self._cal_loss(logits, fg_onehot)
        acc = self._cal_acc(logits, fg_onehot)
        self.log('train_loss', loss, batch_size=bs)
        self.log('train_acc', acc.mean(), batch_size=bs)
        
        return {"loss": loss, "logits": logits}
    
    def validation_step(self,batch, batch_idx):
        spectra = batch["spectra"].to(self.device_)
        formula_ids = batch["formula"].to(self.device_).int()
        fg_onehot = batch["func_groups"].to(self.device_, torch.float)
        bs = spectra.shape[0]
        
        logits, _ =  self(spectra, formula_ids)
        
        loss = self._cal_loss(logits, fg_onehot)
        acc = self._cal_acc(logits, fg_onehot)
        self.log('val_loss', loss, batch_size=bs)
        self.log('val_acc', acc.mean(), batch_size=bs)
        
        return {"loss": loss, "logits": logits}
    
    def test_step(self, batch, batch_idx):
        spectra = batch["spectra"].to(self.device_)
        formula_ids = batch["formula"].to(self.device_).int()
        fg_onehot = batch["func_groups"].to(self.device_, torch.float)
        bs = spectra.shape[0]
        
        logits, _ =  self(spectra, formula_ids)
        
        loss = self._cal_loss(logits, fg_onehot)
        acc = self._cal_acc(logits, fg_onehot)
        self.log('test_loss', loss, batch_size=bs)
        self.log('test_acc', acc.mean(), batch_size=bs)
        
        return {"loss": loss, "logits": logits}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9
            )
        def rate(step):
            if step == 0: step = 1
                
            lr_scale = 1 * (
                self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warm_up_step ** (-1.5))
            )

            return lr_scale
        
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=rate
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    