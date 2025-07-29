# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
import torch.nn as nn


class DistillationLoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, beta: float, tau: float, num_classes: int):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        # Only accepted distillation types after refactoring
        assert distillation_type in ['none', 'soft', 'hard', 'dkd', 'ctkd', 'pkcd']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

        # PKCD internal parameters and buffers
        if distillation_type == 'pkcd':
            self.pkcd_alpha = 1.0
            self.pkcd_beta = 1.0
            self.register_buffer('previous_teacher_logits', None)
            self.register_buffer('previous_teacher_labels', None)
            self.num_classes = num_classes # Number of classes, added for PKCD
        else:
            self.pkcd_alpha = None
            self.pkcd_beta = None
            self.register_buffer('previous_teacher_logits', None) # Still register to avoid error if accessed
            self.register_buffer('previous_teacher_labels', None) # Still register to avoid error if accessed
            self.num_classes = None

        # CTKD specific parameter initialization
        if distillation_type == 'ctkd':
            self.tau_scale = nn.Parameter(torch.tensor(1.0))
            self.register_buffer('base_tau', torch.tensor(tau))
        else:
            self.tau_scale = None
            self.register_buffer('base_tau', None) # Still register to avoid error if accessed

        self.EPS = 1e-8


    def _compute_pearson_correlation(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """
        Computes the Pearson correlation coefficient between two tensors (typically logits or
        probability vectors).
        Assumes input tensors are of shape (Batch_size, Num_classes).
        Computes Pearson correlation coefficient for each sample's two vectors.
        Returns: a tensor of shape (Batch_size,) containing the correlation coefficient for each sample.
        """
        tensor1 = tensor1.to(torch.float32)
        tensor2 = tensor2.to(torch.float32)

        mean1 = tensor1.mean(dim=-1, keepdim=True)
        mean2 = tensor2.mean(dim=-1, keepdim=True)

        centered_tensor1 = tensor1 - mean1
        centered_tensor2 = tensor2 - mean2

        covariance = (centered_tensor1 * centered_tensor2).sum(dim=-1)
        std1 = torch.sqrt((centered_tensor1 ** 2).sum(dim=-1))
        std2 = torch.sqrt((centered_tensor2 ** 2).sum(dim=-1))

        correlation = covariance / (std1 * std2 + self.EPS)
        correlation = torch.clamp(correlation, -1.0, 1.0)
        return correlation

    def _compute_dkd_loss(self, outputs_kd, teacher_outputs, original_labels, T):
        p_teacher = F.softmax(teacher_outputs / T, dim=1)
        p_student = F.softmax(outputs_kd / T, dim=1)

        target_mask = torch.zeros_like(p_teacher).scatter_(
            1,
            original_labels.unsqueeze(1).to(torch.int64),
            1
        )
        
        log_p_student_target = torch.log(p_student + self.EPS)
        log_p_teacher_target = torch.log(p_teacher + self.EPS)

        tckd = F.kl_div(
            log_p_student_target * target_mask,
            log_p_teacher_target * target_mask,
            reduction='sum',
            log_target=True
        ) * (T ** 2) / original_labels.size(0)

        non_target_mask = 1 - target_mask
        
        log_p_student_nontarget = torch.log(p_student + self.EPS)
        log_p_teacher_nontarget = torch.log(p_teacher + self.EPS)

        nckd = F.kl_div(
            log_p_student_nontarget * non_target_mask,
            log_p_teacher_nontarget * non_target_mask,
            reduction='sum',
            log_target=True
        ) * (T ** 2) / original_labels.size(0)

        return tckd + nckd

    def _compute_ctkd_loss(self, outputs_kd, teacher_outputs, T):
        eps = self.EPS
        max_logit = 20.0

        logit_gap = torch.norm(outputs_kd - teacher_outputs, p=2, dim=1).mean()
        scaled_gap = logit_gap / (self.tau_scale.abs() + eps)
        dynamic_T = 1.0 + self.base_tau * torch.sigmoid(scaled_gap)
        dynamic_T = torch.clamp(dynamic_T, min=0.5, max=5.0)

        student_logits = outputs_kd / (dynamic_T + eps)
        teacher_logits = teacher_outputs / (dynamic_T + eps)

        student_logits = torch.clamp(student_logits, min=-max_logit, max=max_logit)
        teacher_logits = torch.clamp(teacher_logits, min=-max_logit, max=max_logit)

        student_log_probs = F.log_softmax(student_logits, dim=1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=1)

        valid_mask = (teacher_log_probs.isfinite() & student_log_probs.isfinite()).all(dim=1)
        if valid_mask.sum() == 0:
            distillation_loss = torch.tensor(1e-5, device=outputs_kd.device)
        else:
            distillation_loss = F.kl_div(
                student_log_probs[valid_mask],
                teacher_log_probs[valid_mask],
                reduction='batchmean',
                log_target=True
            ) * (dynamic_T ** 2)

        return torch.clamp(distillation_loss, max=1e4)

    def _compute_pkcd_loss(self, outputs_kd, teacher_outputs, original_labels, T):
        L_c = torch.tensor([0.0], device=outputs_kd.device)
        L_p = torch.tensor([0.0], device=outputs_kd.device)
        L_cbcd = torch.tensor([0.0], device=outputs_kd.device)
        L_intra = torch.tensor([0.0], device=outputs_kd.device)

        device = outputs_kd.device # Assuming all tensors are on the same device

        if self.previous_teacher_logits is not None and self.previous_teacher_labels is not None:
            current_teacher_logits = teacher_outputs
            current_student_logits = outputs_kd

            prev_teacher_logits = self.previous_teacher_logits.to(device)
            prev_teacher_labels = self.previous_teacher_labels.to(device)

            batch_size, num_classes_actual = outputs_kd.shape

            if prev_teacher_logits.shape[1] != self.num_classes:
                raise ValueError("Previous batch teacher logits have inconsistent number of classes.")

            L_c_list = []
            for i in range(batch_size):
                anchor_student_logit = current_student_logits[i]
                positive_teacher_logit = current_teacher_logits[i]

                negative_teacher_logits_current_batch = []
                if i > 0:
                    negative_teacher_logits_current_batch.append(current_teacher_logits[:i])
                if i < batch_size - 1:
                    negative_teacher_logits_current_batch.append(current_teacher_logits[i+1:])
                
                if len(negative_teacher_logits_current_batch) > 0:
                    negative_teacher_logits_current_batch = torch.cat(negative_teacher_logits_current_batch, dim=0)
                else:
                    negative_teacher_logits_current_batch = torch.empty(0, self.num_classes, device=device, dtype=torch.float32)

                sim_pos = self._compute_pearson_correlation(anchor_student_logit.unsqueeze(0), positive_teacher_logit.unsqueeze(0))
                
                if negative_teacher_logits_current_batch.numel() > 0:
                    sim_neg_current = self._compute_pearson_correlation(anchor_student_logit.unsqueeze(0).expand(negative_teacher_logits_current_batch.shape[0], -1), negative_teacher_logits_current_batch)
                else:
                    sim_neg_current = torch.tensor([], device=device, dtype=torch.float32)

                numerator = torch.exp(sim_pos / T)
                
                if sim_neg_current.numel() > 0:
                    denominator = numerator + torch.sum(torch.exp(sim_neg_current / T))
                else:
                    denominator = numerator
                
                L_c_list.append(-torch.log(numerator / (denominator + self.EPS)))
            
            L_c = torch.stack(L_c_list).mean()


            L_p_list = []
            for i in range(batch_size):
                anchor_student_logit = current_student_logits[i]
                current_label = original_labels[i]

                positive_prev_teacher_logits = prev_teacher_logits[prev_teacher_labels == current_label]
                
                negative_prev_teacher_logits = prev_teacher_logits[prev_teacher_labels != current_label]

                if positive_prev_teacher_logits.numel() > 0:
                    all_prev_logits = torch.cat([positive_prev_teacher_logits, negative_prev_teacher_logits], dim=0)
                    
                    sim_anchor_all_prev = self._compute_pearson_correlation(anchor_student_logit.unsqueeze(0).expand(all_prev_logits.shape[0], -1), all_prev_logits) / T
                    
                    term_sum_log = 0.0
                    for pos_logit in positive_prev_teacher_logits:
                        sim_anchor_pos = self._compute_pearson_correlation(anchor_student_logit.unsqueeze(0), pos_logit.unsqueeze(0)) / T
                        
                        numerator_term = torch.exp(sim_anchor_pos)
                        
                        denominator_term = torch.sum(torch.exp(sim_anchor_all_prev))
                        
                        term_sum_log += -torch.log(numerator_term / (denominator_term + self.EPS))
                    
                    L_p_list.append(term_sum_log / positive_prev_teacher_logits.shape[0])
                else:
                    L_p_list.append(torch.tensor([0.0], device=device))

            L_p = torch.stack(L_p_list).mean()

            L_cbcd = L_c + L_p

            outputs_kd_float = outputs_kd.to(torch.float32)
            teacher_outputs_float = teacher_outputs.to(torch.float32)

            P_s_icd = F.softmax(outputs_kd_float / T, dim=0)
            P_t_icd = F.softmax(teacher_outputs_float / T, dim=0)
            
            L_intra_list = []
            for class_idx in range(num_classes_actual):
                ps_class_vector = P_s_icd[:, class_idx].unsqueeze(0)
                pt_class_vector = P_t_icd[:, class_idx].unsqueeze(0)
                
                corr_val = self._compute_pearson_correlation(ps_class_vector, pt_class_vector).squeeze(0)
                L_intra_list.append(corr_val)
            
            avg_corr_icd = torch.stack(L_intra_list).mean()
            L_intra = 1 - (T ** 2) * avg_corr_icd

        else:
            print("Warning: Previous batch logits/labels not available for PKCD. L_cbcd and L_intra are 0.")
            
        # L_kd is the vanilla soft KD loss calculated in forward
        L_kd = F.kl_div(
            F.log_softmax(outputs_kd / T, dim=1),
            F.log_softmax(teacher_outputs / T, dim=1),
            reduction='sum',
            log_target=True
        ) * (T * T) / outputs_kd.numel()

        distillation_loss = L_kd + self.pkcd_alpha * L_cbcd + self.pkcd_beta * L_intra
        
        # Update historical batch data (save current batch teacher logits and labels at the end of each training step)
        self.previous_teacher_logits = teacher_outputs.detach().clone()
        self.previous_teacher_labels = original_labels.detach().clone()

        return distillation_loss


    def forward(self, inputs, outputs, labels, alignment_loss=None):
        if isinstance(outputs, tuple):
            outputs, outputs_kd, alignment_loss = outputs
        else:
            outputs_kd = outputs
            if alignment_loss is None:
                alignment_loss = torch.tensor(0.0, device=outputs.device)

        original_labels = labels.clone()
        if original_labels.dim() == 2:
            original_labels = original_labels.argmax(dim=1)
        original_labels = original_labels.to(torch.long)

        base_loss = self.base_criterion(outputs, original_labels)

        if self.distillation_type == 'none':
            return base_loss, base_loss, torch.tensor(0.0, device=base_loss.device), alignment_loss

        device = next(self.teacher_model.parameters()).device
        inputs = inputs.to(device)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        teacher_outputs = teacher_outputs.to(outputs_kd.device)

        # batch_size, num_classes_actual = outputs_kd.shape # Not directly needed here
        T = self.tau

        distillation_loss = torch.tensor(0.0, device=outputs_kd.device)

        if self.distillation_type == 'soft':
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        elif self.distillation_type == 'dkd':
            distillation_loss = self._compute_dkd_loss(outputs_kd, teacher_outputs, original_labels, T)
        elif self.distillation_type == 'ctkd':
            distillation_loss = self._compute_ctkd_loss(outputs_kd, teacher_outputs, T)
        elif self.distillation_type == 'pkcd':
            distillation_loss = self._compute_pkcd_loss(outputs_kd, teacher_outputs, original_labels, T)

        # Calculate total loss: includes classification loss and distillation loss
        if self.distillation_type == 'pkcd':
            # For PKCD, L_kd is already included within _compute_pkcd_loss.
            # So, the overall loss is base_loss (L_ce) + total PKCD loss.
            loss = base_loss + distillation_loss
        else:
            loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
            
        return loss, base_loss, distillation_loss, alignment_loss