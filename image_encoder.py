"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from openpyxl.styles.builtins import output

from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector

#这里的D是整条文本转化为token向量之后的的宽度 768，后续在再转化为512来与图片向量统一放到向量库中进行查找
class MultiHeadAttention:#多头注意力的计算与输出
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):#注意力头处理函数：把768维的注意力qkv转化为64*12的注意力头
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].
        H是头的数量，D/H是单个注意力头里面维度的数量
        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """

        # TODO-02 contract: [B,N,D] to [B,H,N,D/H].
        # Image model: D=768, H=12, D/H=64.
        #768=12*64
        x = x.reshape(x.shape[0],x.shape[1],12,64)#shape()提取前两轴的长度
        x = x.transpose([0,2,1,3])#调换顺序
        return x
        raise NotImplementedError('TODO-02: split the feature axis into heads')

    def __call__(self, x, causal=False):#causal=False，前一个token不知道后一个token的信息
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias#QKV 就是 Query / Key / Value 三份向量
        #每个向量占据着最后一个维度width长度的特征信息，q在0->width;w在width->2*width;v在2*width->最后一个
        #Q和K使得同一个token能同时扮演"查询者"和"被查标签"两种角色
        q = self.split_heads(qkv[:, :, :self.width])#Q · Query 查询 我要找什么
        k = self.split_heads(qkv[:, :, self.width:2*self.width])#K · Key 键我的标签是什么
        v = self.split_heads(qkv[:, :, 2*self.width:])#V · Value 值我的实际内容
        """打个比方：注意力就是"每个 token 去资料库里查资料"。Q是查询词（我在找什么），
        K是每份资料的索引标签（我是讲什么的），V是资料正文（真正要读的内容）。
        做法是拿Q和所有K做点积算相似度 → 得scores；TODO - 03a就是这一步："""
        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).#sqrt开方是为了缩放，防止点积过大把 softmax 推向饱和
        #softmax 就是把一组任意大小的分数，转换成一组"非负、加起来等于 1"的权重，之前在多分类模型中用过
        #因为softmax是方差计算，所以如果不除以8就会让最高的那个霸占了1里面几乎所有的分数，其余的接近于0
        #因为一开始是没有8的，但因为和self.qkv_weight.T做了矩阵乘法（抵消了 64 维点积带来的方差膨胀），放大了数据，所以要缩放回来
        #64维就是单头注意力的维度大小
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        scores = (q @ k.swapaxes(-1,-2))/np.sqrt(self.head_dim)
        #[B,H,N,64]->[B,H,N,N]
        #q:[B,H,N,64]  k:[B,H,N,64]
        #如果直接 q @ k：(N,64) @ (N,64) → 内维 64 vs N 对不上（N=50≠64），直接报错
        #所以需要k.swapaxes(-1,-2)将k:[B,H,N,64]->[B,H,64,N]
        #进而q @ k.swap → (B,H,N,64) @ (B,H,64,N) → (B,H,N,N)
        #raise NotImplementedError('TODO-03a: calculate scores from q and k')
        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # 提供的：逐行 softmax，形状仍是 [B,H,N,N]
        #coefficients就是softmax之后的注意力权重，形状[B, H, N, N]，和scores完全一样
        #它由提供的 attention_weights(scores) 算出（内部就是"减最大值 → exp(放进经验函数得到经验值) → 除以行和(所有经验值加在一起)"），保证每一行都是一组和为 1、非负的比例
        #意为当第 i 个 token 要去收集信息时，应该从第 j 个 token 那里取多少比例的内容
        # TODO-03b: mix value rows using coefficients; then join head outputs.
        #coefficients 就是注意力权重矩阵，由 scores 经过 softmax 得来
        context=coefficients@v
        #conficients注意力矩阵不需要一个个遍历乘上Value矩阵算出来加到token上的Value值
        #只需要一次矩阵乘法就可以将每个token向下个token获取value信息的部分算出来填到矩阵里
        # context: [B,H,N,64], merged: [B,N,D].
        merged=context.swapaxes(1,2)
        merged=merged.reshape(merged.shape[0],merged.shape[1], merged.shape[2]*merged.shape[3])
        #raise NotImplementedError('TODO-03b: weighted values and merged heads')

        # TODO-03c: output projection, using out_weight and out_bias.
        merged=merged@self.out_weight.T+self.out_bias#进行一次前向运算，输出output
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        return merged
        #raise NotImplementedError('TODO-03c: project merged attention outputs')


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])#LayerNorm（norm1/norm2）管数值稳定
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        #先看 LayerNorm，它做的事就是沿 token 的特征轴（最后一维 768）把每个 token 各自标准化，再乘上可学习的缩放和偏置
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]【3072，768】
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]【768，3072】
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()#GELU 管引入非线性激活函数，GELU 相比 ReLU 的好处是处处平滑可导

    def __call__(self, x, causal=False):
        """x：进入这个 block 的 token 表示，[B,50,768]。
        第一层来自 pre_norm(embed(images))，之后每层都来自上一个 block 的输出。
         • update：注意力分支算出的更新量，代码里是已提供的 update = self.attention(self.norm1(x))。
         它不是最终结果，只是"注意力建议怎么改这些 token"，形状仍是 [B,50,768]。
         叫 update 就是为了和最终结果区分——你要做的
         TODO-04a 就是把它加到 x 上。
         • residual：残差相加的结果，residual = x + update。
         即"原样保住的信息 + 注意力带来的新信息"。虽然名字叫 residual，
         在代码里它指的是相加之后的主干，后面的 norm2 和第二次相加都基于它。
         • u：没有特殊含义，就是个局部变量名（u = self.norm2(residual)），
         表示送进 MLP 之前的归一化输入。代码里注释写"the second Norm"指的就是它。
         • MLP：多层感知机（前馈网络），对每个 token 各自做 768→3072 升维 → GELU → 3072→768 降维。
        它的输出 mlp_update 再加回 residual，就是本层输出。"""
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        residual=update+x
        #raise NotImplementedError('TODO-04a: add the attention update to x')

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        MLP=self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        MLP=MLP@ self.fc2_weight.T+ self.fc2_bias#两次前向运算
        #两层transformer
        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T  + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        return residual+MLP
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        raise NotImplementedError('TODO-04b: MLP and the second residual')


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):#把图像切成 patch，每个 patch 当成"一个词"，于是图像也变成了一串 token，可以直接套用文本那套 Transformer
        """TODO-01. [B,3,224,224] -> [B,49,3072].BCHW
        #B是Batch不能动
        需要将224*224变成7*32*7*32的小块
        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        images=images.reshape(images.shape[0],3,7,32,7,32)#将224展平为7*32
                            # images.reshape(B, C, H//p, p, W//p, p)
        #其中7的物理意义实际上是沿着224展开成32的7个小单元的编号
        #配凑3072=32*32*3，49=7*7
        images=images.transpose([0,2,4,1,3,5])#其中1是channel就是RGB通道数，3和5则是沿着H*W切成的32*32分辨率小图片
        #而2和4则是patch小批次的意思，不能调换，因为H在前W在后的顺序不能变
        images=images.reshape(images.shape[0],49,3072)#49=7*7共有49个小批次输入，每次输入3*32*32分辨率的小图片
        #reshape(B, (H // p) * (W // p), C * p * p)
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # Input [B,3,224,224], return [B,49,3072].
        return images
        raise NotImplementedError('TODO-01: turn each image into 49 patch rows')

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        patches=patches@self.patch_weight.swapaxes(-1,-2)
        #raise NotImplementedError('TODO-05a: apply the supplied patch projection')

        # Given: one supplied CLS row per image.
        # CLS（Classification token）是一个不对应任何图像内容、纯靠训练学出来的特殊向量
        # 它被插在 token 序列的最前面，专门用来"汇总整张图的信息"
        #CLS 之所以能代表整图，是因为在注意力里它是一个主动查询所有 patch 的 query：scores 矩阵的第 0 行就是 CLS 对 49 个 patch（包括自己）的打分
        # softmax 后加权求和，CLS 这一行就混入了全图信息
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        patches=np.concatenate((cls,patches), axis=1)#拼接成50个patch的BND矩阵,cls和patches千万不能写反
        # Return [B,50,768]. self.position broadcasts over B.
        patches=patches+self.position
        return patches
        #raise NotImplementedError('TODO-05b: CLS, patch rows and positions')

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):#返回每张图的单位向量
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)
        output=tokens[:,0,:]
        output=self.post_norm(output)
        output=output@self.projection
        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already
        # [in,out].[768,512] 已是 in_out，不要转置
        # Name the projected array output. The final unit-vector helper is given.
        #raise NotImplementedError('TODO-06: CLS selection, post-Norm and projection')
        return unit_vector(output)
