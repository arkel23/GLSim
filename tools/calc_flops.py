import os
from decimal import Decimal
import pandas as pd

from glsim import ViTConfig
from glsim.other_utils.build_args import parse_inference_args


class FGFLOPS(object):
    """Computes the inference flops for FGIR transformers."""

    def __init__(self, model_name, image_size=224, patch_size=16, hidden_size=768, 
                 num_classes=1000, channels_in=3, **kwargs):
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.channels_in = channels_in

        self.seq_len = self.calc_seq_len()

        self.num_attention_heads = self.hidden_size // 64

        if self.model_name in ('transfg', 'ffvt'):
            self.num_hidden_layers = 11
        else:
            self.num_hidden_layers = 12

    def calc_seq_len(self):
        if self.model_name in ('transfg', 'aftrans'):
            stride_size = self.patch_size - 4
        else:
            stride_size = self.patch_size
        seq_len = int((((self.image_size - self.patch_size) / stride_size) + 1)) ** 2
        print(seq_len)
        seq_len += 1
        return seq_len

    def vit_flops(self):
        patch_flops = 2 * (self.seq_len - 1) * (self.patch_size ** 2) * self.channels_in * self.hidden_size

        msa_flops = (4 * self.seq_len * (self.hidden_size ** 2)) + (2 * (self.seq_len ** 2) * self.hidden_size)
        pwffn_flops = 8 * self.seq_len * (self.hidden_size ** 2)
        layerwise_flops = msa_flops + pwffn_flops

        out_flops = self.hidden_size * self.num_classes

        flops = patch_flops + (self.num_hidden_layers * layerwise_flops) + out_flops

        return flops
    
    def transfg_flops(self):
        # recursive matrix-matrix multiplication of head-wise attention scores
        # nm * (2p - 1)
        headwise_flops = (self.seq_len ** 2) * ((2 * self.seq_len) - 1)
        # nm * (2p - 1) * num_heads
        layerwise_flops = headwise_flops * self.num_attention_heads
        # nm * (2p - 1) * num_heads * (num_layers - 1)
        flops = layerwise_flops * (self.num_hidden_layers - 1)
        return flops

    def ffvt_flops(self):
        # softmax normalized element-wise first-row and first-column of attention scores
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/profiler/internal/flops_registry.py
        # 2 * n (compute shifted logits) + n (exp of shifted logits) + 2 * n (softmax from exp of shifted logits)
        softmax_flops = 5 * self.seq_len * self.num_attention_heads
        # mean of each head: sum heads and then divide for both a and b vector
        avg_flops = ((self.num_attention_heads - 1) + 1) * 2
        element_wise_prod_flops = self.seq_len
        layerwise_flops = softmax_flops + avg_flops + element_wise_prod_flops
        flops = layerwise_flops * self.num_hidden_layers
        return flops

    def aftrans_flops(self):
        # head-wise element-wise multiplication then sequence-wise GAP
        # followed by excitation-like MLP (2 layers)
        headwise_flops = (self.seq_len ** 2)
        layerwise_flops = (self.num_attention_heads - 1) * headwise_flops

        # squeeze
        gap_flops = (self.seq_len ** 2) + 1

        # excitation: L to C/R dimension (since C/R is not given assume same as L) + L (activation)
        excitation_mlp_layerwise_flops = (self.num_hidden_layers * 2) + self.num_hidden_layers
        # assume select cls tokens only to multiply element-wise excitation of each layer (:, :, 0, 1:)
        excitation_flops = self.num_hidden_layers * (self.seq_len - 1)

        flops = (self.num_hidden_layers * layerwise_flops) + gap_flops + (excitation_mlp_layerwise_flops * 2) + excitation_flops
        return flops

    def ramstrans_flops(self):
        # recursive matrix-matrix multiplication of each layer renormalized attention weights
        # head-wise mean + add diagonal matrix (assume seq_len * seq_len size)
        numerator_flops = (self.num_attention_heads * (self.seq_len ** 2)) + (self.seq_len ** 2)
        # seq-wise (last dimension) mean of head-wise mean (+ diagonal matrix) (already computed in previous step)
        denominator_flops = self.seq_len * (self.seq_len + 1)

        layerwise_norm_flops = numerator_flops + denominator_flops + 1

        # recursive matrix-matrix multiplication
        flops_matrix_mult = (self.seq_len * self.seq_len) * ((2 * self.seq_len) - 1)
        flops = (self.num_hidden_layers * layerwise_norm_flops) + (self.num_hidden_layers - 1) * flops_matrix_mult
        return flops

    def dcal_flops(self):
        # attention rollout:
        # recursive matrix-matrix multiplication of attention weights + identity to account for residual
        # head-wise mean + add diagonal matrix (assume seq_len * seq_len size)
        numerator_flops = (self.num_attention_heads * (self.seq_len ** 2)) + (self.seq_len ** 2)
        # divide all elements by 0.5
        denominator_flops = self.seq_len

        layerwise_norm_flops = numerator_flops + denominator_flops + 1

        # recursive matrix-matrix multiplication
        flops_matrix_mult = (self.seq_len * self.seq_len) * ((2 * self.seq_len) - 1)
        flops = (self.num_hidden_layers * layerwise_norm_flops) + (self.num_hidden_layers - 1) * flops_matrix_mult
        return flops

    def pim_flops(self):
        # weakly supervised selector (linear layer to predict classes for each token) for 4 blocks
        tokenwise_linear_flops = self.hidden_size * self.num_classes
        layerwise_flops = self.seq_len * tokenwise_linear_flops
        flops = 4 * layerwise_flops        
        return flops

    def glsim_flops(self):
        # cosine similarity between CLS token (global) and each token in sequence (local)
        # element-wise multiplication
        numerator_flops = self.hidden_size
        # element-wise squared followed by squared root times 2 (A and B norm each)
        denominator_flops = (self.hidden_size + 1) * 2 
        # add numerator and denominator flops followed by division
        cos_flops = numerator_flops + denominator_flops + 1
        # cosine similarity for each element in sequence
        flops = cos_flops * (self.seq_len - 1)
        return flops

    def get_discriminative_flops(self):
        if self.model_name == 'vit':
            flops = self.vit_flops()
        if self.model_name == 'transfg':
            flops = self.transfg_flops()
        elif self.model_name == 'ffvt':
            flops = self.ffvt_flops()
        elif self.model_name == 'aftrans':
            flops = self.aftrans_flops()
        elif self.model_name == 'ramstrans':
            flops = self.ramstrans_flops()
        elif self.model_name == 'dcal':
            flops = self.dcal_flops()
        elif self.model_name == 'pim':
            flops = self.pim_flops()
        elif self.model_name == 'glsim':
            flops = self.glsim_flops()
        return flops


def main():
    args = parse_inference_args()
    args.results_dir = args.results_inference

    if 'vit_b' in args.model_name:
        hidden_size = 768
    elif 'vit_t' in args.model_name:
        hidden_size = 192

    patch_size = int(args.model_name.split('_')[-1])

    results = []

    method = 'vit'
    flops = FGFLOPS(method, image_size=args.image_size, patch_size=patch_size, num_classes=200,
                    hidden_size=hidden_size).get_discriminative_flops()
    flops = round((flops / 1e6), 2)
    flops_bb = flops
    percent_bb = 100 * (flops / flops_bb)
    print('{}: {:.2f} MFLOPs'.format(method, flops))

    scientific_notation = "%.1E" % Decimal(flops)
    # Split the scientific notation into base and exponent parts
    base, exponent = scientific_notation.split('E')
    # Format the string as desired
    flops_latex = f"${base}\\times 10^{int(exponent)}$"

    results.append({'backbone': args.model_name, 'image_size': args.image_size,
                    'method': method, 'flops': flops, 'flops_latex': flops_latex, 'percent_bb': percent_bb})

    # for method in ('transfg', 'ffvt', 'aftrans', 'ramstrans', 'dcal', 'pim', 'glsim'):
    for method in ('transfg', 'dcal', 'ffvt', 'aftrans', 'glsim'):

        flops = FGFLOPS(method, image_size=args.image_size, patch_size=patch_size,
                        num_classes=200, hidden_size=hidden_size).get_discriminative_flops()
        flops = round((flops / 1e6), 4)
        percent_bb = round(100 * (flops / flops_bb), 4)
        print('{}: {:.2f} MFLOPs'.format(method, flops))

        scientific_notation = "%.1E" % Decimal(flops)
        # Split the scientific notation into base and exponent parts
        base, exponent = scientific_notation.split('E')
        # Format the string as desired
        flops_latex = f"${base}\\times 10^{int(exponent)}$"

        results.append({'backbone': args.model_name, 'image_size': args.image_size,
                        'method': method, 'flops': flops, 'flops_latex': flops_latex, 'percent_bb': percent_bb})

    df = pd.DataFrame.from_dict(results)
    fp = os.path.join(args.results_dir, f'flops_{args.model_name}_{args.image_size}.csv')
    df.to_csv(fp, header=True, index=False)

if __name__ == "__main__":
    main()
