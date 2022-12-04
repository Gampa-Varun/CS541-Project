import tensorflow as tf
import numpy as np

def to_2tuple(x):
    return (x, x)

def DropPath(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (tf.size(x) - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = tfp.distributions.Bernoulli(probs=keep_prob).sample(sample_shape=x.shape)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor/keep_prob

    return x * random_tensor


def window_partition(x, window_size=7):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size (Default: 7)
        
    Returns:
        windows: (num_windows * B, window_size, window_size, C)
                 (8*8*B, 7, 7, C)
    """
    
    B, H, W, C = x.shape
    
    # Convert to (B, 8, 7, 8, 7, C) 
    # x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    
    # Convert to (B, 8, 8, 7, 7, C)
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    
    # Efficient Batch Computation - Convert to (B*8*8, 7, 7, C)
    # windows = windows.view(-1, window_size, window_size, C)
    windows = tf.reshape(windows, [-1, window_size, window_size, C])
    
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows * B, window_size, window_size, C)
                 (8*8*B, 7, 7, C)
        window_size (int): window size (default: 7)
        H (int): Height of image (patch-wise)
        W (int): Width of image (patch-wise)
        
    Returns:
        x: (B, H, W, C)
    """
    
    # Get B from 8*8*B
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    
    # Convert to (B, 8, 8, 7, 7, C)
    # x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])

    # Convert to (B, 8, 7, 8, 7, C)
    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    
    # Convert to (B, H, W, C)
    # x = x.view(B, H, W, -1)
    x = tf.reshape(x,[B, H, W, -1])
    
    return x

class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.keras.activations.gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = tf.keras.layers.Dense(hidden_features, activation=tf.keras.activations.gelu)
        # self.act_layer = act_layer()
        self.fc2 = tf.keras.layers.Dense(out_features, activation=None)
        self.drop = tf.keras.layers.Dropout(drop)
        
    def call(self, x):
        x = self.fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(tf.keras.layers.Layer):
    """ Window based multi-head self attention(W-MSA) module with relative position bias.
        Used as Shifted-Window Multi-head self-attention(SW-MSA) by providing shift_size parameter in
        SwinTransformerBlock module
        
    Args:
        dim (int): Number of input channels (C)
        window_size (tuple[int]): The height and width of the window (M)
        num_heads (int): Number of attention heads for multi-head attention
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v (Default: True)
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight (Default: 0.0)
        proj_drop (float, optional): Dropout ratio of output (Default: 0.0)
    """
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size # Wh(M), Ww(M) (7, 7)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Parameter table of relative position bias: B_hat from the paper
        # (2M-1, 2M-1, num_heads) or (2*Wh-1 * 2*W-1, num_heads)
        self.relative_position_bias_table = tf.Variable(tf.zeros([(2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads]), trainable=True)
        
        # Pair-wise relative position index for each token inside the window
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w)) # (2, M, M) or (2, Wh, Ww)
        coords_flatten = tf.reshape(coords, [2, 49]) # (2, M^2)
        
        # None is dummy dimension
        # coords_flatten[:, :, None] = (2, M^2, 1)
        # coords_flatten[:, None, :] = (2, 1, M^2)
        # relative_coords = (2, M^2, M^2)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        
        # (2, M^2, M^2) -> (M^2, M^2, 2)
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])
        temp = relative_coords.numpy()
        temp[:, :, 0] += self.window_size[0] - 1 # make it start from 0 index
        temp[:, :, 1] += self.window_size[1] - 1
        temp[:, :, 0] *= 2 * self.window_size[1] - 1 # w.r.t x-axis
        relative_coords = tf.convert_to_tensor(temp, dtype=np.int32)
        
        # x-axis + y-axis
        relative_position_index = tf.reduce_sum(relative_coords, -1)
        
        self.relative_position_index = relative_position_index
        
        # Attention
        self.qkv = tf.keras.layers.Dense(dim*3, use_bias=qkv_bias) # W_Q, W_K, W_V
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)
        
        tf.random.truncated_normal(self.relative_position_bias_table.shape, stddev=.02)
        # self.softmax = tf.keras.activations.softmax(axis=-1)
        
    
    def call(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C), N refers to number of patches in a window (M^2)
            mask: (0/-inf) mask with shape of (num_windows, M^2, M^2) or None
                  -> 0 means applying attention, -inf means removing attention
        """
        # (batch, M^2, C)
        B_, N, C = x.shape
        # (num_windows*B, N, 3C)
        qkv = self.qkv(x)
    
        # (B, N, 3, num_heads, C // num_heads)
        # qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, C // self.num_heads])

        # Permute to (3, B_, num_heads, N, C // num_heads)
        '''
        3: referring to q, k, v (total 3)
        B: batch size
        num_heads: multi-headed attention
        N:  M^2, referring to each token(patch)
        C // num_heads: Each head of each of (q,k,v) handles C // num_heads -> match exact dimension for multi-headed attention
        '''
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        # Decompose to query/key/vector for attention
        # each of q, k, v has dimension of (B_, num_heads, N, C // num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2] # Why not tuple-unpacking?
        
        q = q * self.scale
        
        # attn becomes (B_, num_heads, N, N) shape
        # N = M^2

        k = tf.transpose(k, perm=[0, 1, 3, 2])
        attn = (q @ k)
        
        # Remember that relative_position_bias_table = ((2M-1)*(2M-1), num_heads), B_hat from the paper
        # relative_position_index's elements are in range [0, 2M-2]
        # Convert to (M^2, M^2, num_heads). This is B matrix from the paper
        a = tf.reshape(self.relative_position_index, -1)
        b = tf.gather(self.relative_position_bias_table, a)
        a = tf.reshape(b, [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = a
        # Convert to (num_heads, M^2, M^2) to match the dimension for addition
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        
        # (B, num_heads, N, N) + (1, num_heads, M^2, M^2), where N=M^2
        # attn becomes (B_, num_heads, N, N) or (B, num_heads, M^2, M^2)

        attn = attn + tf.expand_dims(relative_position_bias, 0)
        
        if mask is not None:
            nW = mask.shape[0] # nW = number of windows
            
            # attn.view(...) = (B, nW, num_heads, N, N)
            # mask.unsqueeze(1).unsqueeze(0) = (1, num_windows, 1, M^2, M^2)
            # So masking is broadcasted along B and num_heads axis which makes sense
            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N, N]) + tf.expand_dims(tf.expand_dims(mask, 1), 0)
            
            # attn = (nW * B, num_heads, N, N)
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
            attn = tf.keras.activations.softmax(attn)
        else:
            attn = tf.keras.activations.softmax(attn)
            
        attn = tf.keras.activations.softmax(attn)
        
        # attn = (nW*B, num_heads, N, N)
        # v = (B_, num_heads, N, C // num_heads). B_ = nW*B
        # attn @ v = (nW*B, num_heads, N, C // num_heads)
        # (attn @ v).transpose(1, 2) = (nW*B, N, num_heads, C // num_heads)
        # Finally, x = (nW*B, N, C), reshape(B_, N, C) performs concatenation of multi-headed attentions
        x = (attn @ v)
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), [B_, N, C])
        
        # Projection Matrix (W_0). dim doesn't change since we used C // num_heads for MSA
        # x = (B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x        

class SwinTransformerBlock(tf.keras.layers.Layer):
    """ Swin Transformer Block. It's used as either W-MSA or SW-MSA depending on shift_size
    
    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        num_heads (int): Number of attention heads
        window_size (int): Window size
        shift_size (int): Shift size for SW-MSA
        mlp_ratio (float):Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer(nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): NOrmalization layer. Default: nn.LayerNorm
    """
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=tf.keras.activations.gelu, norm_layer=tf.keras.layers.LayerNormalization
                ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        
        # If window_size > input_resolution, no partition
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer()

        # Attention
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        
        if drop_path > 0.:
            self.drop_path_status = True
        else:
            self.drop_path_status = False

        self.norm2 = norm_layer()
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # Attention Mask for SW-MSA
        # This handling of attention-mask is my favourite part. What a beautiful implementation.
        if self.shift_size > 0:
            H, W = self.input_resolution
            
            # To match the dimension for window_partition function
            img_mask = np.zeros([1, H, W, 1])
            
            # h_slices and w_slices divide a cyclic-shifted image to 9 regions as shown in the paper
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            
            # Fill out number for each of 9 divided regions
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
                
            img_mask = tf.convert_to_tensor(img_mask)
                    
            mask_windows = window_partition(img_mask, self.window_size) # (nW, M, M, 1)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
            
            # Such a gorgeous code..
            attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)

            tmask1 = (attn_mask == 0)
            attn_mask = tf.where(tmask1, float(0.0), float(-100.0))

        else:
            attn_mask = None
            
        self.attn_mask = attn_mask
        
    
    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x # Residual
        x = self.norm1(x)
        x = tf.reshape(x, [B, H, W, C]) # H, W refer to the number of "patches" for width and height, not "pixels"
        
        # Cyclic Shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x
        
        # Partition Windows
        x_windows = window_partition(shifted_x, self.window_size) # (nW*B, M, M, C)
        x_windows = tf.reshape(x_windows, [-1, self.window_size*self.window_size, C]) # (nW*B, window_size*window_size, C)
        
        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask) # (nW*B, window_size*window_size, C)
        
        # Merge Windows
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W) # (B, H', W', C)

        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        x = tf.reshape(x, [B, H*W, C])

        # FFn
        x = shortcut + DropPath(x, self.drop_path)
    
        x = x + DropPath(self.mlp(self.norm2(x)), self.drop_path)
        # x = x + (self.mlp(self.norm2(x)))
        
        return x

class PatchMerging(tf.keras.layers.Layer):
    """ Patch Merging Layer from the paper (downsampling)
    Args:
        input_solution (tuple[int]): Resolution of input feature
        dim (int): Number of input channels. (C)
        norm_layer (nn.Module, optional): Normalization layer. (Default: nn.LayerNorm)
    """
    
    def __init__(self, input_resolution, dim, norm_layer=tf.keras.layers.LayerNormalization):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = tf.keras.layers.Dense(2 * dim)
        self.norm = norm_layer()

        
    def call(self, x):
        """
        x: (B, H*W, C)
        """
        
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = tf.reshape(x, [B, H, W, C])
        
        # Separate per patch by 2 x 2
        t = x.numpy()
        x0 = t[:, 0::2, 0::2, :] # (B, H/2, W/2, C) (top-left of 2x2)
        x1 = t[:, 1::2, 0::2, :] # (B, H/2, W/2, C) (bottom-left of 2x2)
        x2 = t[:, 0::2, 1::2, :] # (B, H/2, W/2, C) (top-right of 2x2)
        x3 = t[:, 1::2, 1::2, :] # (B, H/2, W/2, C) (bottom-right of 2x2)
        
        # Merge by channel -> (B, H/2, W/2, 4C)
        x = tf.concat([x0, x1, x2, x3], -1)
        
        # Flatten H, W
        x = tf.reshape(x, [B, -1, 4 * C])
        
        x = self.norm(x)
        
        # Reduction Layer: 4C -> 2C
        x = self.reduction(x)
        
        return x

class BasicLayer(tf.keras.layers.Layer):
    """ Swin Transformer layer for one stage
    
    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        depth (int): Number of blocks (depending on Swin Version - T, L, ..)
        num_heads (int): Number of attention heads
        window_size (int): Local window size
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. (Default: True)
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop (float, optional): Dropout rate (Default: 0.0)
        attn_drop (float, optional): Attention dropout rate (Default: 0.0)
        drop_path (float | tuple[float], optional): Stochastic depth rate (Default: 0.0)
        norm_layer (nn.Module, optional): Normalization layer (Default: nn.LayerNorm)
        downsample (nn.Module | NOne, optional): Downsample layer at the end of the layer (Default: None)
        use_checkpoint (bool): Whether to use checkpointing to save memory (Default: False)
    """
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=tf.keras.layers.LayerNormalization, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # Build  Swin-Transformer Blocks
        self.blocks = tf.keras.Sequential([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio = mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer
                                )
            for i in range(depth)
        ])
        
        
        # Patch Merging Layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
            
            
    def call(self, x):
        self.blocks(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x

class PatchEmbed(tf.keras.layers.Layer):
    """ Convert image to patch embedding
    
    Args:
        img_size (int): Image size (Default: 224)
        patch_size (int): Patch token size (Default: 4)
        in_channels (int): Number of input image channels (Default: 3)
        embed_dim (int): Number of linear projection output channels (Default: 96)
        norm_layer (nn.Module, optional): Normalization layer (Default: None)
    """
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) # (img_size, img_size) to_2tuple simply convert t to (t,t)
        patch_size = to_2tuple(patch_size) # (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]] # (num_patches, num_patches)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # proj layer: (B, 3, 224, 224) -> (B, 96, 56, 56)
        self.proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)
        
        if norm_layer is not None:
            self.norm = norm_layer()
        else:
            self.norm = None
        
    def call(self, x):
        """
        x: (B, C, H, W) Default: (B, 3, 224, 224)
        returns: (B, H//patch_size * W//patch_size, embed_dim) (B, 56*56, 96)
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}]) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
 
        # (B, 3, 224, 224) -> (B, 96, 56, 56)
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        x = self.proj(x)
        x = tf.transpose(x, perm=[0, 3, 2, 1])

        # (B, 96, 56, 56) -> (B, 96, 56*56)
        x = tf.reshape(x, [x.shape[0], x.shape[1], x.shape[2]*x.shape[3]])

        # (B, 96, 56*56) -> (B, 56*56, 96): 56 refers to the number of patches
        x = tf.transpose(x, perm=[0, 2, 1])

        if self.norm is not None:
            x = self.norm(x)

        return x

class SwinTransformer(tf.keras.layers.Layer):
    """ Swin Transformer
    
    Args:
        img_size (int | tuple(int)): Input image size (Default 224)
        patch_size (int | tuple(int)): Patch size (Default: 4)
        in_chans (int): Number of input image channels (Default: 3)
        num_classes (int): Number of classes for classification head (Default: 1000)
        embed_dim (int): Patch embedding dimension (Default: 96)
        depths (tuple(int)): Depth of each Swin-T layer
        num_heads (tuple(int)): Number of attention heads in different layers
        window_size (int): Window size (Default: 7)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. (Default: 4)
        qkv_bias (bool): If True, add a learnable bias to query, key, value (Default: True)
        qk_scale (float); Override default qk scale of head_dim ** -0.5 if set. (Default: None)
        drop_rate (float): Dropout rate (Default: 0)
        attn_drop_rate (float): Attention dropout rate (Default: 0)
        drop_path_rate (float); Stochastic depth rate (Default: 0.1)
        norm_layer (nn.Module): Normalization layer (Default: nn.LayerNorm)
        ape (bool): Refers to absolute position embedding. If True, add ape to the patch embedding (Default: False)
        patch_norm (bool): If True, add normalization after patch embedding (Default: True)
        use_checkpoint (bool): Whether to use checkpointing to save memory (Default: False)
    """
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=tf.keras.layers.LayerNormalization, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        self.pos_drop = tf.keras.layers.Dropout(drop_rate)
        
        # Stochastic Depth
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))] # stochastic depth decay rule
        
        # build layers
        self.layers = tf.keras.Sequential(
        [BasicLayer(
            dim=int(embed_dim * 2 ** i_layer),
            input_resolution=(
                patches_resolution[0] // (2 ** i_layer), # After patch-merging layer, patches_resolution(H, W) is halved
                patches_resolution[1] // (2 ** i_layer),
                                ),
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging if (i_layer < self.num_layers -1) else None, # No patch merging at the last stage
            use_checkpoint=use_checkpoint
        ) for i_layer in range(self.num_layers)])
            
        self.norm = norm_layer()
        self.avgpool = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first", keepdims=True)
        
        # Classification Head
        self.head = tf.keras.layers.Dense(num_classes) if num_classes > 0 else tf.identity
        
    def forward_features(self, x):
        x = self.patch_embed(x)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)

        x = self.norm(x) # (B, L, C)
        print("x", x.shape)
        t = tf.transpose(x, perm=[0, 2, 1])
        print("t", t.shape)
        xavg = self.avgpool(t) # (B, C, 1)
        print("xavg", xavg.shape)
        #x = torch.flatten(x, 1)
        return x, xavg
    
    def call(self, x):
        x, xavg = self.forward_features(x)
        return x, xavg
    
if __name__ == '__main__':

    import tensorflow as tf
    import tensorflow_probability as tfp

    model = SwinTransformer()
    dummy = np.random.rand(10, 3, 224, 224)
    
    print(model(dummy)[0].shape)
    print(model(dummy)[1].shape)