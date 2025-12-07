import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_24
from fvcore.nn import flop_count
from utils.flops import give_supported_ops
import copy
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
import torch
from modeling.clip import clip
from modeling.fusion_part.STR_CHI import STR_CHI
from utils.simple_tokenizer import SimpleTokenizer


class STMI(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(STMI, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = self.feat_dim
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss_type = cfg.TEST.MISS
        self.mask = cfg.MODEL.MASK
        self.probability = self.cfg.MODEL.PROBABILITY
        self.STR_CHI = cfg.MODEL.STR_CHI
        self.learn_tokens = cfg.MODEL.LEARNABLE_TOKENS
        self.threshold = cfg.MODEL.THRESHOLD

        if self.STR_CHI:
            self.STR_CHI = STR_CHI(self.feat_dim, self.learn_tokens, self.threshold)
            self.visual_classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.visual_classifier.apply(weights_init_classifier)
            self.bottleneck_visual = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_visual.bias.requires_grad_(False)
            self.bottleneck_visual.apply(weights_init_kaiming)

            self.textual_classifier = nn.Linear(3*self.feat_dim, self.num_classes, bias=False)        # 
            self.textual_classifier.apply(weights_init_classifier)
            self.bottleneck_textual = nn.BatchNorm1d(3*self.feat_dim)
            self.bottleneck_textual.bias.requires_grad_(False)
            self.bottleneck_textual.apply(weights_init_kaiming)

            print('~~~~~~~~~~~~~~~Using STR_CHI~~~~~~~~~~~~~~~')
        if self.direct:
            self.classifier_v = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_v.apply(weights_init_classifier)
            self.bottleneck_v = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_v.bias.requires_grad_(False)
            self.bottleneck_v.apply(weights_init_kaiming)

            self.classifier_t = nn.Linear(self.feat_dim, self.num_classes, bias=False)    #
            self.classifier_t.apply(weights_init_classifier)
            self.bottleneck_t = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t.bias.requires_grad_(False)
            self.bottleneck_t.apply(weights_init_kaiming)
        else:
            self.classifier_t = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_t.apply(weights_init_classifier)
            self.bottleneck_t = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t.bias.requires_grad_(False)
            self.bottleneck_t.apply(weights_init_kaiming)

            self.classifier_v_nir = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_v_nir.apply(weights_init_classifier)
            self.bottleneck_v_nir = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_v_nir.bias.requires_grad_(False)
            self.bottleneck_v_nir.apply(weights_init_kaiming)

            self.classifier_v_tir = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_v_tir.apply(weights_init_classifier)
            self.bottleneck_v_tir = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_v_tir.bias.requires_grad_(False)
            self.bottleneck_v_tir.apply(weights_init_kaiming)

            self.classifier_v_rgb = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_v_rgb.apply(weights_init_classifier)
            self.bottleneck_v_rgb = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_v_rgb.bias.requires_grad_(False)
            self.bottleneck_v_rgb.apply(weights_init_kaiming)

        self.tokenizer = SimpleTokenizer()

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def flops(self, shape=(3, 256, 128)):
        if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
            shape = (3, self.image_size[0], self.image_size[1])
            # For vehicle reid, the input shape is (3, 128, 256)
        supported_ops = give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
        input_r = torch.ones((1, *shape), device=next(model.parameters()).device, dtype=torch.float32)
        input_n = torch.ones((1, *shape), device=next(model.parameters()).device, dtype=torch.float32)
        input_t = torch.ones((1, *shape), device=next(model.parameters()).device, dtype=torch.float32)
        input_m = torch.ones((1, 1, 16, 8), device=next(model.parameters()).device, dtype=torch.float32)
        cam_label = torch.tensor(0, device=next(model.parameters()).device, dtype=torch.int64)
        input = {"RGB": input_r, "NI": input_n, "TI": input_t, 'mask':input_m, "cam_label": cam_label,
                 'text': {'rgb_text': clip.tokenize('just a test.').cuda(),
                          'ni_text': clip.tokenize('just a test.').cuda(),
                          'ti_text': clip.tokenize('just a test.').cuda()}}
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "The out_proj here is called by the nn.MultiheadAttention, which has been calculated in th .forward(), so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("For the bottleneck or classifier, it is not calculated during inference, so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "For the Mamba Series, the code implementations are all used with the inner weight instead of directly calling the model, the FLOPs has been calculated with our inner function 'MambaInnerFn_jit', so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        del model, input
        return sum(Gflops.values()) * 1e9

    def forward(self, image, text=None, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None,
                writer=None, epoch=None):
        if 'cam_label' in image:
            cam_label = image['cam_label']
        if 'text' in image:
            Text = image['text']['rgb_text']
        else:
            Text = text['rgb_text']
        real_text = []
        for i in range(len(Text)):
            real_text.append(self.tokenizer.decode(Text[i].tolist()))
        text_real = {'real_text': real_text}
        if self.training:
            RGB = image['RGB']
            NI = image['NI']
            TI = image['TI']
            if self.mask:
                mask = image['mask']
            else:
                mask = None

            RGB_v_feas, RGB_v_global = self.BACKBONE(image=RGB, text=None,
                                                                               cam_label=cam_label, label=label,
                                                                               view_label=view_label, use_mask=mask, prob = self.probability)
            NI_v_feas, NI_v_global = self.BACKBONE(image=NI, text=None, cam_label=cam_label,
                                                                           label=label,
                                                                           view_label=view_label, use_mask=mask, prob = self.probability)
            TI_v_feas, TI_v_global = self.BACKBONE(image=TI, text=None, cam_label=cam_label,
                                                                           label=label,
                                                                           view_label=view_label, use_mask=mask, prob = self.probability)

            t_fea, t_global = self.BACKBONE.encode_text(Text)
            
            if self.STR_CHI:
                boss_fea = torch.stack([RGB_v_global, NI_v_global, TI_v_global, t_global],
                                       dim=1)
                visual = self.STR_CHI(RGB_v_feas, NI_v_feas, TI_v_feas, boss_fea)
                score_vv = self.visual_classifier(self.bottleneck_visual(visual))
            if self.direct:
                ori_v = torch.cat([RGB_v_global, NI_v_global, TI_v_global], dim=-1)
                ori_t = t_global           
                score_v = self.classifier_v(self.bottleneck_v(ori_v))
                score_t = self.classifier_t(self.bottleneck_t(ori_t))
                if self.STR_CHI:
                    return score_v, ori_v, score_t, ori_t, score_vv, visual
                else:
                    return score_v, ori_v, score_t, ori_t
            else:
                score_t = self.classifier_t(self.bottleneck_t(t_global))
                score_rgb_v = self.classifier_v_rgb(self.bottleneck_v_rgb(RGB_v_global))
                score_nir_v = self.classifier_v_nir(self.bottleneck_v_nir(NI_v_global))
                score_tir_v = self.classifier_v_tir(self.bottleneck_v_tir(TI_v_global))
                if self.STR_CHI:
                    return score_rgb_v, RGB_v_global, score_nir_v, NI_v_global, score_tir_v, TI_v_global, \
                        score_t, t_global, score_vv, visual
                else:
                    return score_rgb_v, RGB_v_global, score_nir_v, NI_v_global, score_tir_v, TI_v_global, \
                        score_t, t_global

        else:
            RGB = image['RGB']
            NI = image['NI']
            TI = image['TI']
            if self.mask:
                mask = image['mask']
            else:
                mask = None
            if self.miss_type == 'r':
                RGB = torch.zeros_like(RGB)
            elif self.miss_type == 'n':
                NI = torch.zeros_like(NI)
            elif self.miss_type == 't':
                TI = torch.zeros_like(TI)
            elif self.miss_type == 'rn':
                RGB = torch.zeros_like(RGB)
                NI = torch.zeros_like(NI)
            elif self.miss_type == 'rt':
                RGB = torch.zeros_like(RGB)
                TI = torch.zeros_like(TI)
            elif self.miss_type == 'nt':
                NI = torch.zeros_like(NI)
                TI = torch.zeros_like(TI)

            NI_v_feas, NI_v_global = self.BACKBONE(image=NI, text=None, cam_label=cam_label,
                                                                           view_label=view_label, use_mask=mask, prob = 0.0)
            RGB_v_feas, RGB_v_global = self.BACKBONE(image=RGB, text=None,
                                                                               cam_label=cam_label,
                                                                               view_label=view_label, use_mask=mask, prob = 0.0)
            TI_v_feas, TI_v_global = self.BACKBONE(image=TI, text=None, cam_label=cam_label,
                                                                           view_label=view_label, use_mask=mask, prob = 0.0)

            t_fea, t_global = self.BACKBONE.encode_text(Text)

            multi_modal_dict = {"V_RGB": RGB_v_global, "V_NIR": NI_v_global, "V_TIR": TI_v_global, "T": t_global,
                                'LOCAL_v': torch.cat([RGB_v_global, NI_v_global, TI_v_global], dim=-1),
                                'LOCAL': torch.cat(
                                    [RGB_v_global, NI_v_global, TI_v_global, t_global],
                                    dim=-1)}
            if self.STR_CHI:
                boss_fea = torch.stack([RGB_v_global, NI_v_global, TI_v_global, t_global],
                                       dim=1)
                visual = self.STR_CHI(RGB_v_feas, NI_v_feas, TI_v_feas, boss_fea)
                multi_modal_dict['LOCAL'] = visual
            return multi_modal_dict

__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_24': t2t_vit_t_24,
}

def make_model(cfg, num_class, camera_num, view_num=0):
    model = STMI(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building STMI===========')
    return model
