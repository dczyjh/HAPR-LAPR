import copy
import json
import time
from pathlib import Path
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.data import DataManager
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES
from tqdm import tqdm
import math

from clip.model import VisionTransformer, convert_weights
from .efficient_adaptation import adaptation_parameter, configure_student_adaptation
from tools.provenance import training_identity
from .adaptation_optim import adaptation_param_groups
from .adaptation_diagnostics import AdaptationDiagnostics
from .development import base_development_dataset

_tokenizer = _Tokenizer()

class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )
    def forward(self, input_feat):
        
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))
        
        return final_feat.squeeze(-1).squeeze(-1)
        
def load_clip_to_cpu_teacher(cfg, zero_shot_model=False):
    backbone_name = cfg.TRAINER.PROMPTKD.TEACHER_NAME
    # url = clip._MODELS[backbone_name]
    
    if backbone_name == "ViT-B/16":
        model_path = str(Path(cfg.TRAINER.PROMPTKD.WEIGHTS_ROOT) / 'ViT-B-16.pt')
    elif backbone_name == "ViT-L/14":
        model_path = str(Path(cfg.TRAINER.PROMPTKD.WEIGHTS_ROOT) / 'ViT-L-14.pt')
    elif backbone_name == "ViT-B/32":
        model_path = str(Path(cfg.TRAINER.PROMPTKD.WEIGHTS_ROOT) / 'ViT-B-32.pt')
    else:
        print('enter the wrong teacher name.')
    
    print(f"CLIP Teacher name is {backbone_name}")
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # We default use PromptSRC to pretrain our teacher model
    design_details = {"trainer": 'IVLP',
                        "vision_depth": 9,
                        "language_depth": 9,
                        "vision_ctx": 4,
                        "language_ctx": 4}
    
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    # url = clip._MODELS[backbone_name]
    model_path = str(Path(cfg.TRAINER.PROMPTKD.WEIGHTS_ROOT) / 'ViT-B-16.pt')
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.PROMPTKD.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.PROMPTKD.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        
        # print(f'------prompts size is {prompts.size()}------')
        # print(f'------tokenized prompts size is {tokenized_prompts.size()}------')

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, is_teacher):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTKD.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTKD.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        self.trainer_name = cfg.TRAINER.NAME
        self.train_modal = cfg.TRAINER.MODAL
        
        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTKD.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        
        print(f'classnames size is {len(classnames)}')

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # self.name_lens = name_lens

        if self.train_modal == "base2novel":
            self.register_buffer("token_prefix", embedding[:math.ceil(self.n_cls / 2), :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:math.ceil(self.n_cls / 2), 1 + n_ctx:, :])  # CLS, EOS

            self.register_buffer("token_prefix2", embedding[math.ceil(self.n_cls / 2):, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[math.ceil(self.n_cls / 2):, 1 + n_ctx:, :])  # CLS, EOS
            
        elif self.train_modal == "cross":
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            
            self.register_buffer("token_prefix2", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        # print(f'label is {label}')
        # if label is not None:
        #     prefix = prefix[label]
        #     suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        # print(f'ctx size is {ctx.size()}')

        prefix = self.token_prefix
        # print(f'prefix size is {prefix.size()}')
        
        suffix = self.token_suffix
        # print(f'suffix size is {suffix.size()}')

        if self.trainer_name == "PromptKD" and self.train_modal == "base2novel":
            # print(f'n_cls is {self.n_cls}')
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)

        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        # Additional random draws must not change projector initialization or
        # the data-loader RNG stream in paired baseline/adapted runs.
        with torch.random.fork_rng(devices=[]):
            self.adaptation_summary = configure_student_adaptation(cfg, self.image_encoder)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        
        self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 768)
       
        self.cfg = cfg
        
        if self.dtype == torch.float16:
            convert_weights(self.VPT_image_trans)
        else:
            self.VPT_image_trans.to(dtype=self.dtype)

    def forward(self, image, label=None, return_intermediates=False, intermediate_layers=None):
        logit_scale = self.logit_scale.exp()

        encoder_output = self.image_encoder(
            image.type(self.dtype),
            return_intermediates=return_intermediates,
            intermediate_layers=intermediate_layers,
        )
        if return_intermediates:
            image_features, intermediates = encoder_output
        else:
            image_features = encoder_output
            intermediates = None
        image_features = self.VPT_image_trans(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if return_intermediates:
            return image_features, logit_scale, intermediates
        return image_features, logit_scale


class CustomCLIP_teacher(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, True)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cached_text_features = None

    @torch.no_grad()
    def get_text_features(self):
        """Return the frozen teacher class vectors, computing them once."""
        if self.cached_text_features is None:
            prompts = self.prompt_learner()
            device = self.logit_scale.device
            text_features = self.text_encoder(
                prompts.to(device), self.tokenized_prompts.to(device)
            )
            self.cached_text_features = (
                text_features / text_features.norm(dim=-1, keepdim=True)
            ).detach()
        return self.cached_text_features

    def forward(self, image=None, label=None):
        text_features = self.get_text_features()
        
        logit_scale = self.logit_scale.exp()
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        
        logits = logit_scale * image_features @ text_features.t()
        
        return image_features, text_features, logits


@TRAINER_REGISTRY.register()
class PromptKD(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTKD.PREC in ["fp16", "fp32", "amp"]
        if cfg.TRAINER.PROMPTKD.DEVELOPMENT and (
                cfg.TRAINER.MODAL != "base2novel" or cfg.TEST.FINAL_MODEL != "last_step"):
            raise ValueError("Development requires base2novel and fixed last checkpoint.")
        if cfg.TRAINER.PROMPTKD.DIAGNOSTICS.INTERVAL < 0 or cfg.TRAINER.PROMPTKD.DIAGNOSTICS.PROBE_SIZE < 1:
            raise ValueError("Invalid diagnostics interval or probe size.")

    def build_data_loader(self):
        transform = base_development_dataset if self.cfg.TRAINER.PROMPTKD.DEVELOPMENT else None
        dm = DataManager(self.cfg, dataset_transform=transform)
        for name in ("train_loader_x", "train_loader_u", "val_loader", "test_loader",
                     "num_classes", "num_source_domains", "lab2cname"):
            setattr(self, name, getattr(dm, name))
        self.dm = dm

    def build_model(self):
        cfg = self.cfg
        
        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        clip_model_teacher = load_clip_to_cpu_teacher(cfg)

        if cfg.TRAINER.PROMPTKD.PREC == "fp32" or cfg.TRAINER.PROMPTKD.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            clip_model_teacher.float()

        self.rep_weight = float(cfg.TRAINER.PROMPTKD.ADAPTATION.REP_WEIGHT)
        self.rep_layers = tuple(int(layer) for layer in cfg.TRAINER.PROMPTKD.ADAPTATION.REP_LAYERS)
        self.reference_encoder = None
        if self.rep_weight > 0:
            # Copy before HARP/LARP injection. The frozen copy supplies a true
            # pre-adaptation semantic reference, including the same initial VPT.
            self.reference_encoder = copy.deepcopy(clip_model.visual)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        print(f"Student adaptation: {self.model.adaptation_summary}")

        self.model_teacher = CustomCLIP_teacher(cfg, classnames, clip_model_teacher)
        
        if cfg.TRAINER.MODAL == "base2novel":
            model_path = str(Path(cfg.TRAINER.PROMPTKD.TEACHER_ROOT) / cfg.DATASET.NAME / 'VLPromptLearner/model-best.pth.tar')
        elif cfg.TRAINER.MODAL == "cross":
            model_path = str(Path(cfg.TRAINER.PROMPTKD.TEACHER_ROOT) / 'ImageNet-xd/VLPromptLearner_large/model.pth.tar-20')
            
        self.train_modal = cfg.TRAINER.MODAL
        
        self.teacher_path = model_path
        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]
        
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]
        if "prompt_learner.token_prefix2" in state_dict:
            del state_dict["prompt_learner.token_prefix2"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]
        if "prompt_learner.token_suffix2" in state_dict:
            del state_dict["prompt_learner.token_suffix2"]
        
        incompatible = self.model_teacher.load_state_dict(state_dict, strict=False)
        allowed_missing = {"prompt_learner.token_prefix", "prompt_learner.token_suffix",
                           "prompt_learner.token_prefix2", "prompt_learner.token_suffix2"}
        missing = set(incompatible.missing_keys) - allowed_missing
        unexpected = [key for key in incompatible.unexpected_keys
                      if not key.startswith("prompt_learner.ZS_image_encoder.")]
        if missing or unexpected:
            raise RuntimeError(f"Teacher checkpoint mismatch: missing={missing}, "
                               f"unexpected={unexpected}")
        self.model_teacher.to(self.device)
        self.model_teacher.eval()
        self.model_teacher.requires_grad_(False)
        # PromptKD's paper pre-stores the frozen teacher class vectors. The
        # released code recomputes them every batch; cache them here without
        # changing the logits.
        self.model_teacher.get_text_features()
        
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name or adaptation_parameter(name):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        trainable_parameter_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                trainable_parameter_count += param.numel()
        print(f"Parameters to be updated: {enabled}")
        print(f"Trainable tensor count: {len(enabled)}")
        print(f"Trainable scalar parameter count: {trainable_parameter_count}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        if self.reference_encoder is not None:
            self.reference_encoder.to(self.device)
            self.reference_encoder.eval()
            for parameter in self.reference_encoder.parameters():
                parameter.requires_grad_(False)
        # NOTE: only give prompt_learner to the optimizer

        self.trainable_list = nn.ModuleList([])
        self.trainable_list.append(self.model)

        self.optim = build_optimizer(self.trainable_list, cfg.OPTIM,
                                     param_groups=adaptation_param_groups(self.model, cfg))
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTKD.PREC == "amp" else None
        self.amp_skipped_steps = 0
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.temperature = cfg.TRAINER.PROMPTKD.TEMPERATURE
        self.run_identity, self.comparison_signature = training_identity(
            cfg, self.dm.dataset, self.teacher_path
        )
        self.trainable_parameter_count = trainable_parameter_count
        self.diagnostics = AdaptationDiagnostics(self.model, cfg, self.output_dir)
        self.global_step = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def after_train(self):
        """Evaluate Base and Novel on exactly the same saved checkpoint."""
        print("Finish training")
        if not self.cfg.TEST.NO_TEST:
            epoch = None if self.cfg.TEST.FINAL_MODEL == "best_val" else self.max_epoch
            self.load_model(self.output_dir, epoch=epoch)
            development = self.cfg.TRAINER.PROMPTKD.DEVELOPMENT
            base = float(self.test(split="val"))
            novel = None if development else float(self.test(split="test"))
            result = {
                "schema_version": 3, "status": "complete",
                "evaluation_role": "development_base" if development else "benchmark",
                "method": self.cfg.TRAINER.PROMPTKD.ADAPTATION.TYPE,
                "dataset": self.cfg.DATASET.NAME, "seed": self.cfg.SEED,
                "kd_weight": float(self.cfg.TRAINER.PROMPTKD.KD_WEIGHT),
                "rep_weight": self.rep_weight, "epochs": self.max_epoch,
                "precision": self.cfg.TRAINER.PROMPTKD.PREC,
                "selection": self.cfg.TEST.FINAL_MODEL,
                "selected_epoch": self.loaded_epoch,
                "elapsed_seconds": time.time() - self.time_start,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                "teacher_path": str(Path(self.teacher_path).resolve()),
                "torch_version": torch.__version__,
                "comparison_signature": self.comparison_signature,
                "identity": self.run_identity,
                "trainable_parameters": self.trainable_parameter_count,
                "amp_skipped_steps": self.amp_skipped_steps,
                "adaptation": self.model.module.adaptation_summary if isinstance(self.model, nn.DataParallel) else self.model.adaptation_summary,
                "adaptation_spec": self.cfg.TRAINER.PROMPTKD.ADAPTATION.dump(),
            }
            if development:
                result["base_development"] = base
                result["development_source"] = "original_val_base_only"
            else:
                result.update(base=base, novel=novel,
                              hm=2 * base * novel / (base + novel) if base + novel else 0.)
            output = Path(self.output_dir) / ("development_metrics.json" if development else "metrics.json")
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(output)
        self.close_writer()

    def _compute_student_loss(self, image, tea_text_features, tea_logits):
        use_representation_loss = self.reference_encoder is not None and self.rep_weight > 0
        student_output = self.model(
            image,
            return_intermediates=use_representation_loss,
            intermediate_layers=self.rep_layers,
        )
        if use_representation_loss:
            image_ft, logit_scale, student_intermediates = student_output
        else:
            image_ft, logit_scale = student_output
            student_intermediates = {}

        stu_logits = logit_scale * image_ft @ tea_text_features.t().detach()
        raw_kd_loss = F.kl_div(
            F.log_softmax(stu_logits / self.temperature, dim=1),
            F.softmax(tea_logits / self.temperature, dim=1),
            reduction='sum',
        ) * (self.temperature * self.temperature) / stu_logits.numel()
        kd_loss = self.cfg.TRAINER.PROMPTKD.KD_WEIGHT * raw_kd_loss

        representation_loss = kd_loss.new_zeros(())
        if use_representation_loss:
            model_dtype = self.model.module.dtype if isinstance(self.model, nn.DataParallel) else self.model.dtype
            with torch.no_grad():
                _, reference_intermediates = self.reference_encoder(
                    image.type(model_dtype),
                    return_intermediates=True,
                    intermediate_layers=self.rep_layers,
                )

            per_layer_losses = []
            for layer_index in self.rep_layers:
                if layer_index not in student_intermediates or layer_index not in reference_intermediates:
                    raise RuntimeError(f"Missing representation at layer {layer_index}")
                student_feature = F.normalize(student_intermediates[layer_index].float(), dim=-1)
                reference_feature = F.normalize(
                    reference_intermediates[layer_index].detach().float(), dim=-1
                )
                per_layer_losses.append(
                    1.0 - F.cosine_similarity(student_feature, reference_feature, dim=-1).mean()
                )
            representation_loss = torch.stack(per_layer_losses).mean()

        total_loss = kd_loss + self.rep_weight * representation_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite loss: stop before corrupting the checkpoint.")
        return total_loss, kd_loss, representation_loss
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        with torch.no_grad():
            tea_image_features, tea_text_features, tea_logits = self.model_teacher(image)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.PROMPTKD.PREC
        diagnostics = getattr(self, "diagnostics", None)
        self.global_step = getattr(self, "global_step", 0) + 1
        skipped = False
        if prec == "amp":
            with autocast():
                loss, kd_loss, representation_loss = self._compute_student_loss(
                    image, tea_text_features, tea_logits
                )
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            if diagnostics is not None:
                diagnostics.before_step(self.global_step, image, optim)
            old_scale = scaler.get_scale()
            scaler.step(optim)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            self.amp_skipped_steps += int(skipped)
        else:
            loss, kd_loss, representation_loss = self._compute_student_loss(
                image, tea_text_features, tea_logits
            )
            optim.zero_grad()
            loss.backward()
            # A finite loss does not imply finite FP16 gradients. Stop before
            # SGD writes NaNs/Infs into a checkpoint; never silently clip them.
            gradient_norms = [p.grad.detach().float().norm()
                              for p in model.parameters() if p.grad is not None]
            if gradient_norms and not torch.isfinite(torch.stack(gradient_norms)).all():
                raise FloatingPointError("Non-finite gradient before optimizer step. "
                                         "Inspect the run; compare all methods in AMP if needed.")
            if diagnostics is not None:
                diagnostics.before_step(self.global_step, image, optim)
            optim.step()

        if diagnostics is not None:
            diagnostics.after_step(self.global_step, tea_text_features, skipped)

        loss_summary = {
            "loss": loss.item(),
            "loss_kd": kd_loss.item(),
            "loss_rep": representation_loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            self.loaded_epoch = int(epoch)

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_prefix2" in state_dict:
                del state_dict["prompt_learner.token_prefix2"]
                
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            if "prompt_learner.token_suffix2" in state_dict:
                del state_dict["prompt_learner.token_suffix2"]
                
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        if self.cfg.TRAINER.PROMPTKD.DEVELOPMENT and split != "val":
            raise ValueError("Development mode only permits Base development evaluation (split='val').")
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)

            tea_text_features = self.model_teacher.get_text_features()
                
            image_ft, logit_scale = self.model(image, label)
            
            if self.train_modal == "base2novel":
                if split == "val":
                    output = logit_scale * image_ft @ tea_text_features[:math.ceil(self.n_cls / 2),:].t()
                elif split == "test":
                    output = logit_scale * image_ft @ tea_text_features[math.ceil(self.n_cls / 2):,:].t()
            elif self.train_modal == "cross" :
                output = logit_scale * image_ft @ tea_text_features.t()
            
            self.evaluator.process(output, label) 

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
