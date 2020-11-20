import torch
import torch.nn as nn
from core.model.task_basemodel.backbone.base_model import base_module
from .votenet import votenet
from .scanrefer.loss_helper import get_loss
from .scanrefer.module.lang_module import LangModule
from .scanrefer.module.match_module import MatchModule
from .scanrefer.eval import eval_ref, eval_det

class refernet(base_module):
    def __init__(self, config):
        super(refernet, self).__init__()
        self.votenet = votenet(config.votenet_config)
        refer_type = config.refer_config.name
        self.refer_type = refer_type
        self.refer_config = config.refer_config
        self.eval_reference = True
        self.eval_detection = True
        if refer_type == 'none':
            pass
        elif refer_type == '2stage':
            self.eval_detection = False
            args = {
                "force": True,
                "seed": 42,
                "repeat": 1,
                "no_nms": False,
                "no_lang_cls": False,
                "use_color": False,
                "use_normal": False,
                "use_bidir": False,
                'folder': '',
                'use_oracle': False,
                'use_cat_rand': False,
                'use_best': False
            }
            args.update(self.refer_config)
            self.refer_config.update(args)
            # SCANREFER CONFIG-DEFAULT
            # def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
            # input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
            # use_lang_classifier=True, use_bidir=False, no_reference=False,
            # emb_size=300, hidden_size=256):
            
            # VOTENET CONFIG-DEFAULT
            # self.net = VoteNet(num_class=DATASET_CONFIG.num_class,
            #                    num_heading_bin=DATASET_CONFIG.num_heading_bin,
            #                    num_size_cluster=DATASET_CONFIG.num_size_cluster,
            #                    mean_size_arr=DATASET_CONFIG.mean_size_arr,
            #                    num_proposal=config.num_target,
            #                    input_feature_dim=config.num_input_channel,
            #                    vote_factor=config.vote_factor,
            #                    sampling=config.cluster_sampling)

            use_lang_classifier = not self.refer_config.get('no_lang_cls', False)
            use_bidir = self.refer_config.get('use_bidir', False)
            emb_size = self.refer_config.get('emb_size', 300)
            hidden_size = self.refer_config.get('hidden_size', 256)
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(self.votenet.DATASET_CONFIG.num_class,
                                   use_lang_classifier,
                                   use_bidir,
                                   emb_size,
                                   hidden_size)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(config.votenet_config.num_target,
                                     lang_size=(1 + int(use_bidir)) * hidden_size)
        else:
            raise NotImplementedError(refer_type)

    def _forward(self, input):
        data_dict = self.votenet._forward(input)
        for key in input:
            assert(key not in data_dict)
            data_dict[key] = input[key]
        if self.refer_type == 'none':
            pass
        elif self.refer_type == '2stage':
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################
            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang(data_dict)
            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################
            # --------- PROPOSAL MATCHING ---------
            data_dict = self.match(data_dict)
        return data_dict

    def calculate_loss(self, input, output):
        # print('CALCULATE LOSS!!!')
        loss, output = get_loss(output, self.votenet.DATASET_CONFIG)
        fin_out = {}
        for key, value in output.items():
            # if isinstance(value, torch.Tensor):
            #     print('cal loss', key, value.shape)
            # else:
            #     print('cal loss not tensor', key, value)
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                # print('cal loss --- real loss', key, value)
                if 'loss' not in key:
                    key = key + '_loss'
                fin_out[key] = value.detach()
        fin_out['loss'] = loss
        return fin_out

    def individual_tester(self, loader, transform_input, test_freq):
        args = self.votenet.DATASET_CONFIG
        finval = None
        if self.eval_reference:
            acc_ref = eval_ref(self.refer_config, self, loader, args)
            finval = 1 - acc_ref
        if self.eval_detection:
            acc_det = eval_det(self.refer_config, self, loader, args)
            if finval is None:
                finval = 1 - acc_det
        return finval
