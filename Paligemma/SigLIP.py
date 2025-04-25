from typing import Optional, Tuple
import torch
import torch.nn as nn


class SiglipVisionConfig:

    def __init__(
            self, 
            hidden_size = 768,
            intermediate_size = 3072,
            num_hidden_layers = 12,
            num_attention_heads = 12,
            num_channels = 3,
            image_size = 224,
            patch_size = 16,
            layer_norm_eps = 1e-6,
            attention_dropout = 0.0,
            num_image_tokens: int = None,
            *args, **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    """Transforms an image to multiple patches with embeddings; furthermore, positional embeddings are also added to establish 
    and understand relationships between patches"""

    def __init__(self, config: SiglipVisionConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates padding is unnecessary 
        )


        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor: 
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        # Convolve the 'patch_size' kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // 
        """In a general case, the convolution formula is: new_H = (H - (kernel-1)- 1)/stride + 1"""
        patch_embeds = self.patch_embedding(pixel_values)

        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)

        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1,2)

        # Add positional embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings += self.position_embedding(self.position_ids)

        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
    

class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention is All You Need' paper """

    def __init__(self, config: SiglipVisionConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** (-0.5)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        query_states = self.q_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (query_states @ key_states.transpose(-1, -2)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be size {(batch_size, self.num_heads, seq_len, seq_len)}, but is" 
                f" {attn_weights.size()}"
            )
        
        # Apply the softmax row-wise.
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout during training.
        attn_weights = nn.functional.dropout(input=attn_weights, p=self.dropout, training=self.training)

        # [Batch_Size, Num_Heads, Num_Patches, Num_Patches] -> [Batch_Size, Num_Heads, Num_Patches, Embed_Dim]
        attn_output = (attn_weights @ value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.embed_dim):
            raise ValueError(
                f"Attention output should be size {(batch_size, self.num_heads, seq_len, self.embed_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights



class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # Non-Linear activation function
        hidden_states = nn.functional.gelu(input=hidden_states, approximate='tanh')
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim] 
        hidden_states = self.fc2(hidden_states)

        return hidden_states



class SiglipEncoderLayer(nn.Module):
    """Singular layer of the Transformer Encoder block"""

    def __init__(self, config: SiglipVisionConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual = [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # Run through LayerNorm: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # Run through Multi-head attention: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # Add in residual connection
        hidden_states = residual + hidden_states

        # Save residual : [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # Run through LayerNorm: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # Run through MLP: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # Add in residual connection 
        hidden_states = residual + hidden_states

        return hidden_states



class SiglipEncoder(nn.Module):
    """List of layers of a transformer: resembles the encoder side of the original transformer."""

    def __init__(self, config: SiglipVisionConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        # input_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = input_embeds

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        
        return hidden_states


class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Shape: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values=pixel_values)

        last_hidden_state = self.encoder(inputs_embeds = hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModule(nn.Module):

    def __init__(self, config: SiglipVisionConfig, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # Shape: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values) 
    """ Can also be written as self.vision_model.__call__(pixel_values) which then calls the forward function of SiglipVisionTransformer"""
    
