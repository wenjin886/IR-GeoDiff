import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import copy

from .spec_utils import (Embeddings, 
                         EmbedPatchAttention, 
                         PositionalEncoding, 
                         Encoder, EncoderLayer,
                         Decoder, DecoderLayer,
                         MultiHeadedAttention,
                         PositionwiseFeedForward,
                         )

class AtomsEmbed(nn.Module):
    def __init__(self, d_model, atoms_vocab_size) -> None:
        super(AtomsEmbed, self).__init__()
        self.d_model = d_model

        self.embed = Embeddings(d_model=self.d_model, vocab=atoms_vocab_size)
        # self.pe = PositionalEncoding(d_model=d_model, dropout=0.1)
        self.pe_ = PositionalEncoding(d_model=self.d_model, dropout=0)
        self.atom_dropout = nn.Dropout(p=0.1)


    def forward(self, atoms):
        atoms_ = self.embed(atoms)

        # Group PE
        atoms_pe_ = self.atom_pe(atoms)
        atoms_ = self.atom_dropout(atoms_ + atoms_pe_)

        return atoms_
    
    def atom_pe(self, atoms):
        """
        Only apply positional encoding on the same type of atoms.
        """
        pe_atoms = []
        for mol_atoms in atoms:
            unique_atoms = torch.unique(mol_atoms)
            grouped_indices = [torch.where(mol_atoms == token)[0] for token in unique_atoms]
            pe_mol_atoms = torch.zeros(len(mol_atoms), self.d_model).to(atoms.device)
            for indices in grouped_indices:
                pe_mol_atoms[indices] = self.pe_(pe_mol_atoms[indices].unsqueeze(0)).squeeze(0)
            pe_atoms.append(pe_mol_atoms)
        return torch.stack(pe_atoms)
    

        

class SpecAtomEncoder(nn.Module):
    def __init__(self, config):
        super(SpecAtomEncoder, self).__init__()

        self.device_ = config.device
        d_model = config.d_model
        atoms_vocab_size = config.atoms_vocab_size

        c = copy.deepcopy
        pe = PositionalEncoding(d_model, dropout=0.1)

        self.atoms_embed = AtomsEmbed(d_model=d_model, atoms_vocab_size=atoms_vocab_size)
        
        self.spec_len = config.spec_len
        patch_len = config.patch_len
        self.spec_patch_num = int(self.spec_len/patch_len)
        spec_patch_att = EmbedPatchAttention(spec_len=self.spec_len, 
                                              patch_len=patch_len, 
                                              src_vocab=config.spec_vocab,
                                              d_model=d_model,
                                              h=config.n_spec_attention_head)
        self.spec_embed = nn.Sequential(spec_patch_att, c(pe))
        
        encoder_attn = MultiHeadedAttention(h=config.n_spec_atom_encoder_head,d_model=d_model)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=4*d_model, dropout=0.1)
        self.spec_atom_encoder = Encoder(EncoderLayer(d_model, c(encoder_attn),c(ff),0.1), 
                                         N=config.n_spec_atom_encoder_layer)
        decoder_att = MultiHeadedAttention(h=config.n_decoder_head, d_model=d_model,
                                           dropout=config.decoder_dropout)
        self.decoder = Decoder(DecoderLayer(size=d_model, self_attn=c(decoder_att), src_attn=c(encoder_attn), feed_forward=c(ff), dropout=0.1),
                                      N=config.n_decoder_layer)
        
        
        self.__init_weights__()
    
    

    def encode(self, atom_ids, spectra):
        atoms_ = self.atoms_embed(atom_ids)
        spectra_ = self.spec_embed(spectra)

        src_ = torch.cat((spectra_, atoms_), dim=1).to(self.device_)
        src_mask, atoms_mask = self._get_src_mask(atom_ids)
        # print(src_mask.shape, atoms_mask.shape)
        # print(src_.device, src_mask.device)
        src_ = self.spec_atom_encoder(src_, src_mask)
        atoms_ = src_[:, self.spec_patch_num:, :]
        return src_, src_mask, atoms_, atoms_mask
        
    def decode(self, atoms_embedding, atoms_mask, spec_atom_encout, src_mask):
        tgt_mask = atoms_mask 
        return self.decoder(atoms_embedding, memory=spec_atom_encout,src_mask=src_mask, tgt_mask=tgt_mask)
    

    def forward(self, num_atoms, spectra, atom_ids):
        max_num_atoms = max(num_atoms)
        atom_ids = atom_ids[:, :max_num_atoms]

        spec_atoms_encout, src_mask, atoms_embedding, atoms_mask = self.encode(atom_ids, spectra)
        heavy_atoms_embedding = atoms_embedding[:, :max_num_atoms, :]

        dec_out = self.decode(heavy_atoms_embedding, atoms_mask, spec_atoms_encout, src_mask)

        return dec_out, atoms_mask
    
    
    def _get_src_mask(self, atom_ids):
        """
        No padding mask for spec, only for atoms' token.
        """
        (batch_size, _) =  atom_ids.shape

        spectra_att_mask = torch.ones(batch_size, self.spec_patch_num, dtype=torch.bool).to(self.device_) # (B, S)
        atoms_mask = (atom_ids != 0).to(self.device_) 
        src_mask = torch.cat((spectra_att_mask, atoms_mask), dim=-1).unsqueeze(-2)  # (B, 1, S+N)

        return src_mask, atoms_mask.unsqueeze(-2)
    
    def _subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask == 0
    
    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                init.xavier_normal_(m.weight)

class SpecEncoder(nn.Module):
    def __init__(self, device="cuda",
                 d_model=512,
                 spec_len=3200,
                 patch_len=64,
                 spec_vocab=100,
                 n_spec_attention_head=8,
                 n_spec_encoder_head=8,
                 n_spec_encoder_layer=4,
                 use_formula=False,
                 formula_vocab_size=26,
                 ):
        super().__init__()

        self.device_ = torch.device(device)
        c = copy.deepcopy
        
        self.spec_len = spec_len
        self.spec_patch_num = int(self.spec_len/patch_len)
        spec_patch_att = EmbedPatchAttention(spec_len=self.spec_len, 
                                              patch_len=patch_len, 
                                              src_vocab=spec_vocab,
                                              d_model=d_model,
                                              h=n_spec_attention_head)
        pe = PositionalEncoding(d_model, dropout=0.1)
        self.spec_embed = nn.Sequential(spec_patch_att, pe)

        self.use_formula = use_formula
        if self.use_formula:
            self.formula_embed = nn.Sequential(Embeddings(d_model=d_model, vocab=formula_vocab_size), c(pe))
        
        encoder_attn = MultiHeadedAttention(h=n_spec_encoder_head,d_model=d_model)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=4*d_model, dropout=0.1)
        self.spec_encoder = Encoder(EncoderLayer(d_model, c(encoder_attn),c(ff),0.1), 
                                         N=n_spec_encoder_layer)
    
        self.__init_weights__()
    

    def forward(self, spectra, formula_ids=None):
        bs = spectra.shape[0]
        spectra_ = self.spec_embed(spectra)
        
        if self.use_formula:
            formula_ = self.formula_embed(formula_ids)
            src_ = torch.cat((spectra_, formula_), dim=1)
            src_mask = self._get_mask(batch_size=bs, formula_ids=formula_ids)
        else:
            src_ = spectra_
            src_mask = self._get_mask(batch_size=bs)
        
        enc_out = self.spec_encoder(src_, src_mask)
        return enc_out, src_mask
    
    def _get_mask(self, batch_size, formula_ids=None):
        """
        No padding mask for spec, only for atoms' token.
        """
        spectra_att_mask = torch.ones(batch_size, self.spec_patch_num, dtype=torch.bool).to(self.device_) # (B, S)

        if self.use_formula:
            formula_att_mask = (formula_ids != 0).to(self.device_) 
            src_mask = torch.cat((spectra_att_mask, formula_att_mask), dim=-1).unsqueeze(-2)  # (B, 1, S+N)
            return src_mask
        else:
            spectra_att_mask = spectra_att_mask.unsqueeze(-2).to(self.device_)  # (B, 1, S)
            return spectra_att_mask
    
    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                init.xavier_normal_(m.weight)

class SpecAtomCrossAttention(nn.Module):
    def __init__(self, device="cuda",
                 d_atom=2,
                 d_model=512,
                 n_head=8,
                 n_layer=4,
                 n_spec_encoder_head=8,
                 spec_patch_num=50,
                 ):
        super().__init__()

        self.device_ = torch.device(device)
        d_model = d_model

        c = copy.deepcopy
        self.atom_proj_1 = nn.Linear(d_atom, d_model)
        
        self.spec_patch_num = spec_patch_num
        spec_attn = MultiHeadedAttention(h=n_spec_encoder_head, d_model=d_model)
        
        decoder_att = MultiHeadedAttention(h=n_head, d_model=d_model, dropout=0.1)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=4*d_model, dropout=0.1)
        self.decoder = Decoder(
            DecoderLayer(size=d_model, self_attn=c(decoder_att), src_attn=c(spec_attn), feed_forward=c(ff), dropout=0.1),
            N=n_layer)
        
        self.atom_proj_2 = nn.Sequential(
            nn.Linear(d_model, int(d_model/2)),
            nn.SiLU(),
            nn.Linear(int(d_model/2), d_atom))
  
        self.__init_weights__()

    def forward(self, atoms, atoms_mask, spectra_embed, spectra_att_mask=None):
        batch_size = spectra_embed.shape[0]
        atoms = atoms.view(batch_size, -1, atoms.shape[-1])
        atoms_mask = atoms_mask.view(batch_size, -1)
        atoms = self.atom_proj_1(atoms)
        atoms_att_mask = atoms_mask.unsqueeze(-2)  # (B, 1, N)
        
        if spectra_att_mask is None:
            spectra_att_mask = torch.ones(batch_size, self.spec_patch_num, dtype=torch.bool).to(self.device_) # (B, S)
            spectra_att_mask = spectra_att_mask.unsqueeze(-2)  # (B, 1, S)
        
        atoms = self.decoder(atoms, spectra_embed, spectra_att_mask, atoms_att_mask)
        atoms = self.atom_proj_2(atoms)
        return atoms
        # return atoms.view(batch_size, -1, atoms.shape[-1])
    
    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                init.xavier_normal_(m.weight)

class SpecEdgeCrossAttention(nn.Module):
    def __init__(self, device="cuda",
                 d_atom=2,
                 d_model=512,
                 n_head=8,
                 n_layer=4,
                 n_spec_encoder_head=8,
                 spec_patch_num=50,
                 edge_output_dim=16,
                 use_fg=False,
                 fg_n_layer=4,
                 use_edge_cross_attn=True
                 ):
        super().__init__()

        self.device_ = torch.device(device)

        c = copy.deepcopy
        self.atom_proj = nn.Linear(d_atom, d_model)
        self.edge_proj_1 = nn.Linear(n_head, d_model)

        self.spec_patch_num = spec_patch_num
        spec_attn = MultiHeadedAttention(h=n_spec_encoder_head, d_model=d_model)
        
        att = MultiHeadedAttention(h=n_head, d_model=d_model, dropout=0.1)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=4*d_model, dropout=0.1)
        
        self.use_edge_cross_attn = use_edge_cross_attn
        if self.use_edge_cross_attn:
            self.cross_attn_with_spec = Decoder(DecoderLayer(size=d_model, self_attn=c(att), src_attn=c(spec_attn), feed_forward=c(ff), dropout=0.1),
                                        N=n_layer)
        else:
            self.self_attn_edges = Encoder(EncoderLayer(d_model, c(att), c(ff), 0.1), 
                                         N=n_layer)
            
        self.edge_proj_2 = nn.Linear(d_model, edge_output_dim)

        assert (d_model % n_head) == 0, f"Please check d_model ({d_model}) and n_head ({n_head})"
        self.n_head = n_head
        self.d_k = int(d_model/n_head)
        
        self.use_fg = use_fg
        if self.use_fg:
            self.cross_attn_with_fg = Decoder(DecoderLayer(size=d_model, src_attn=c(att), feed_forward=c(ff), self_attn=None, dropout=0.1), 
                                              N=fg_n_layer)
        self.__init_weights__()

    
    def forward(self, atoms, atoms_mask, spectra_embed, spectra_att_mask=None, fg_embed=None):
        """
        Args:
            atoms: [x, h], (B, N, F)
            atoms_mask: (B, N, 1)
            spectra_embed: (B, S, D)
        Returns:
            restored_edges: (B, N, N, D_out)
        """
        batch_size, n_nodes, _ = atoms.shape

    
        # 1) Do all the linear projections in batch from d_model => B x H x N x d_k
        atoms = self.atom_proj(atoms).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        

        # 2) Apply attention on all the projected vectors in batch.
        scores = self.get_attention_score(atoms) # B x H x N x N
        scores = scores.permute(0, 2, 3, 1) # B x N x N x H
        score_mask = torch.matmul(atoms_mask, atoms_mask.view(batch_size, 1, n_nodes)) # B x N x N

        # 3) Get edges
        edges, edge_mask, triu_indices = self.get_edges(n_nodes, scores, score_mask)
        edges = self.edge_proj_1(edges)
    
        # 4）Self-attention or Cross-attention with spec tokens
        edge_att_mask = edge_mask.unsqueeze(-2)  # (B, 1, E)
        if self.use_edge_cross_attn:
            if spectra_att_mask is None:
                spectra_att_mask = torch.ones(batch_size, self.spec_patch_num, dtype=torch.bool).to(self.device_) # (B, S)
                spectra_att_mask = spectra_att_mask.unsqueeze(-2)  # (B, 1, S)
            edges = self.cross_attn_with_spec(edges, spectra_embed, spectra_att_mask, edge_att_mask)
        else:
            edges = self.self_attn_edges(edges, edge_att_mask)
        
        
        if self.use_fg:
            assert fg_embed is not None, "Please check the input of fg_enmbed."
            fg_att_mask = torch.ones(batch_size, fg_embed.size(1), dtype=torch.bool).to(self.device_) # (B, FG)
            fg_att_mask = fg_att_mask.unsqueeze(-2)  # (B, 1, FG)
            edges = self.cross_attn_with_fg(edges, fg_embed, fg_att_mask, edge_att_mask)
            
        edges = self.edge_proj_2(edges)

        # 4) Reverse to adj shape
        restored_edges = self.reverse_2_adj(n_nodes, edges, triu_indices)
      
        return restored_edges
    
    def get_attention_score(self, atoms):
        "Compute 'Scaled Dot Product Attention', attention_score"
        # Make scores a symmetric matrix
        scores = torch.matmul(atoms, atoms.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, H, N, N)
        return scores
    
    def get_edges(self, n_nodes, scores, score_mask):
        """
        Args:
            scores: (B, N, N, H)
            score_mask: (B, N, N)
        Returns:
            edges: (B, N*(N-1)/2), the upper triangle matrix of attention score matrix (explcuding diag) 
        """
        triu_indices = torch.triu_indices(n_nodes, n_nodes, offset=1)
        edges = scores[:, triu_indices[0], triu_indices[1], :]
        edge_mask = score_mask[:, triu_indices[0], triu_indices[1]]
        return edges, edge_mask, triu_indices
    
    def reverse_2_adj(self, n_nodes, edges, triu_indices):
        """
        Args:
            edges: (B, N_edges, D_out)
        Returns:
            restored_edges: (B, N, N, D_out)
        """
        restored_edges = torch.zeros(edges.size(0), n_nodes, n_nodes, edges.size(2)).to(edges.device)

        restored_edges[:, triu_indices[0], triu_indices[1], :] = edges
        restored_edges[:, triu_indices[1], triu_indices[0], :] = edges

        return restored_edges
            

    
    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                init.xavier_normal_(m.weight)

    
