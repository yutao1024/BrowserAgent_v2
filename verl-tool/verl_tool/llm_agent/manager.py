import torch
import os
import re
import ray
import uuid
import json
import random
import regex as re
import numpy as np
import requests
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.utils import hf_tokenizer
from verl.utils.model import get_generation_config
from tqdm import tqdm
from typing import List, Union
from .config import AgentActorConfig
from .tensor_helper import TensorHelper, TensorConfig

# 1) A sanitizer that strips all embedded NULs (and, optionally, any
#    other C0 control characters except common whitespace).
CONTROL_CHAR_RE = re.compile(
    # this matches U+0000 through U+001F, excluding tab(09), LF(0A), CR(0D)
    r'[\x00-\x08\x0B\x0C\x0E-\x1F]'
)

def sanitize_request(obj: Any) -> Any:
    """
    Recursively walk through obj and:
      - For dicts: sanitize each value
      - For lists/tuples: sanitize each element
      - For strings: remove embedded nulls (and other control chars)
      - Leave other types untouched
    """
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {sanitize_request(key): sanitize_request(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_request(item) for item in obj)
    elif isinstance(obj, str):
        # strip NUL (\x00) and other C0 control chars
        return CONTROL_CHAR_RE.sub('', obj)
    else:
        return obj

class AgentActorManager:
    def __init__(
        self,
        model_path,
        actor_rollout_wg,
        config: AgentActorConfig,
        is_validation: bool = False,
    ):
        self.model_path = model_path
        self.tokenizer = hf_tokenizer(self.model_path)
        self.generation_config = get_generation_config(self.model_path)
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        # self.logger = logger
        self.is_validation = is_validation
        self.eos_token_id = self.generation_config.eos_token_id \
            if self.generation_config is not None else self.tokenizer.eos_token_id
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length,
            max_response_length=config.max_response_length,
        ))
        if self.config.action_stop_tokens is not None:
            if os.path.exists(self.config.action_stop_tokens):
                with open(self.config.action_stop_tokens, 'r') as f:
                    self.action_stop_tokens = [x for x in f.read().split(',') if x]
                print(f"Using action stop tokens: {self.action_stop_tokens}")
            else:
                raise ValueError(f"action_stop_tokens file not found: {self.config.action_stop_tokens}")
        else:
            self.action_stop_tokens = []
        self.additional_eos_token_ids = self.config.additional_eos_token_ids
        if isinstance(self.additional_eos_token_ids, str):
            self.additional_eos_token_ids = [int(x) for x in self.additional_eos_token_ids.split(',')]
        elif isinstance(self.additional_eos_token_ids, list):
            self.additional_eos_token_ids = [int(x) for x in self.additional_eos_token_ids]
        elif self.additional_eos_token_ids is None:
            self.additional_eos_token_ids = []
        if self.config.mtrl_sep is None:
            messages = [{"role": "system", "content": "{obs}"}]
            self.config.mtrl_sep = "\n" + self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            self.config.mtrl_sep = self.config.mtrl_sep.replace("system", self.config.mtrl_role)
    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors='pt',
            padding="longest"
        )['input_ids']

    def _preprocess_inputs(self, inputs: DataProto):
        """
        this version verl do not repeat the input by n times, so we manually repeat the input by n times
        """
        # we manually repeat the input by n times if needed since every trajectory is independent
        do_sample = inputs.meta_info.get("do_sample", True)
        assert 'traj_ids' in inputs.non_tensor_batch, "traj_ids should be claimed univerally in the ray trainer"
        ori_len = len(inputs.batch['input_ids'])
        if not do_sample:
            n = 1
        else:
            n = self.config.n
            inputs = inputs.repeat(n, interleave=True)
        # 新增：extra_info 也重复 n 倍
        if 'extra_info' in inputs.non_tensor_batch and inputs.non_tensor_batch['extra_info'] is not None:
            ori_extra = inputs.non_tensor_batch['extra_info']
            new_extra = []
            for i in range(ori_len):
                for j in range(n):
                    new_extra.append(ori_extra[i])
            # 保持类型一致
            if isinstance(ori_extra, np.ndarray):
                inputs.non_tensor_batch['extra_info'] = np.array(new_extra, dtype=object)
            else:
                inputs.non_tensor_batch['extra_info'] = new_extra
        # add "_{i}" for each trajectory to the traj_ids
        for i in range(ori_len):
            for j in range(n):
                inputs.non_tensor_batch['traj_ids'][i*n+j] += f"_{j}"
        return inputs

    def _postprocess_responses(self, responses: torch.Tensor, action_step: int) -> torch.Tensor:
        """Process responses to stop at python operation or answer operation."""
        
        effective_lens = self.tensor_fn.create_attention_mask(responses).sum(dim=1)
        do_actions = []
        if self.config.enable_mtrl:
            responses_str = [self.tokenizer.decode(responses[i][:effective_lens[i]], skip_special_tokens=False) for i in range(responses.shape[0])]
            for i in range(len(responses_str)):
                if action_step >= self.config.min_turns:
                    if self.action_stop_tokens:
                        if any([action_stop_token in responses_str[i] for action_stop_token in self.action_stop_tokens]):
                            do_action = True
                            # replace other action stop tokens with the first one
                            for j in range(1, len(self.action_stop_tokens)):
                                if self.action_stop_tokens[j] in responses_str[i]:
                                    responses_str[i] = responses_str[i].replace(self.action_stop_tokens[j], self.action_stop_tokens[0])
                            if not responses_str[i].endswith(self.config.turn_end_token):
                                responses_str[i] += self.config.turn_end_token
                        else:
                            do_action = False
                    else:
                        do_action = True
                else:
                    # always do action, decided by the server about whether an action stops
                    for j in range(1, len(self.action_stop_tokens)):
                        if self.action_stop_tokens[j] in responses_str[i]:
                            responses_str[i] = responses_str[i].replace(self.action_stop_tokens[j], self.action_stop_tokens[0])
                    turn_end_token_idx = responses_str[i].rfind(self.config.turn_end_token)
                    if self.action_stop_tokens and not self.action_stop_tokens[0] in responses_str[i]:
                        if turn_end_token_idx != -1:
                            responses_str[i] = responses_str[i][:turn_end_token_idx] + self.action_stop_tokens[0] + self.config.turn_end_token
                        else:
                            responses_str[i] = responses_str[i] + self.action_stop_tokens[0] + self.config.turn_end_token
                    else:
                        if turn_end_token_idx == -1:
                            responses_str[i] += self.config.turn_end_token
                    do_action = True
                do_actions.append(do_action)
        else:
            responses_str = self.tokenizer.batch_decode(
                responses,
                skip_special_tokens=True
            )
            for i, resp in enumerate(responses_str):
                # resp = resp.strip(' \n')
                has_action = False
                for j in range(len(self.action_stop_tokens)):
                    if self.action_stop_tokens[j] in resp:
                    # if resp.endswith(self.action_stop_tokens[j]):
                    # if self.action_stop_tokens[j] in resp[-(len(self.action_stop_tokens[j]) + 3):]: # 5 for some action token tokens not indepdently decoded
                        has_action = True
                        responses_str[i] = resp.split(self.action_stop_tokens[j])[0] + self.action_stop_tokens[j]
                        break
                if not has_action and action_step < self.config.min_turns:
                    has_action = True
                    responses_str[i] = resp + self.action_stop_tokens[0]
                do_actions.append(has_action)
            for i in range(len(responses_str)):
                if not do_actions[i]:
                    responses_str[i] = self.tokenizer.decode(responses[i][:effective_lens[i]], skip_special_tokens=False) # preserve eos token
        # with open(f"temp-{action_step}.json", 'w') as f:
        #     json.dump([{
        #         "responses_str": responses_str[i],
        #         "do_action": do_actions[i],
        #     } for i in range(len(responses_str))], f, indent=4)
        responses = self._batch_tokenize(responses_str).to(torch.int64)
        return responses, responses_str, do_actions
    
    # def _postprocess_responses(self, responses: torch.Tensor, action_step: int, eos_token_id: Union[list, List[int]]=None) -> torch.Tensor:
    #     """Process responses to stop at python operation or answer operation."""
    #     if not eos_token_id:
    #         eos_token_id = self.eos_token_id
    #     if isinstance(eos_token_id, int):
    #         eos_token_id = [eos_token_id]
    #     eos_token_id += self.additional_eos_token_ids
    #     full_len = responses.shape[1]
    #     effective_lens = self.tensor_fn.create_attention_mask(responses).sum(dim=1)
    #     max_len = effective_lens.max()
    #     responses = responses[:, :max_len]
    #     responses_str = [self.tokenizer.decode(responses[i][:effective_lens[i]], skip_special_tokens=True) for i in range(responses.shape[0])]

    #     if action_step < self.config.min_turns:
    #         # re-encode remove special tokens like eos
    #         responses = self._batch_tokenize(responses_str).to(torch.int64)
    #         # force do action for those effective len not equal to full len
    #         do_actions = [effective_lens[i] != full_len for i in range(len(responses_str))]
    #     else:
    #         do_actions = [
    #             not (responses[i, effective_lens[i]-1] in eos_token_id or effective_lens[i] == full_len) for i in range(responses.shape[0])
    #         ] # consider stop (not do action) when meeting any eos token or the response meet the longest response length. 

    #         for i in range(responses.shape[0]):
    #             if do_actions[i]:
    #                 resp = responses_str[i]
    #                 # sometimes the model can generate pad_token as one of the eos token, then we check if it did not stop with any action stop tokens above, 
    #                 # this is also a finished sequence
    #                 if not any([action_stop_token in resp[-(len(action_stop_token)+3):] for action_stop_token in self.action_stop_tokens]):
    #                     do_actions[i] = False
        
    #     # apply self.config.max_action_length
    #     if self.config.max_action_length is not None and self.config.max_action_length > 0:
    #         responses = responses[:, :self.config.max_action_length]
    #     return responses, responses_str, do_actions

    def _process_next_obs(self, next_obs: List[str], dones: List[bool], valid_action: List[bool], finishs: List[bool]) -> torch.Tensor:
        """Process next observations from environment."""
        mtrl_sep = self.config.mtrl_sep
        next_obs = [obs if not done else "" for obs, done in zip(next_obs, dones)]
        if self.config.truncate_obs_side == 'left':
            next_obs_ids = self.tokenizer(
                next_obs,
                padding='longest',
                return_tensors='pt',
                add_special_tokens=False,  # Prevents adding special tokens
                padding_side='left',
            )['input_ids'].to(torch.int64)
            if next_obs_ids.shape[1] > self.config.max_obs_length:
                print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")
                next_obs_ids = next_obs_ids[:, -self.config.max_obs_length:]
        elif self.config.truncate_obs_side == 'right':
            next_obs_ids = self.tokenizer(
                next_obs,
                padding='longest',
                return_tensors='pt',
                add_special_tokens=False,  # Prevents adding special tokens
                padding_side='right',
            )['input_ids'].to(torch.int64)
            if next_obs_ids.shape[1] > self.config.max_obs_length:
                print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")
                next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
        else:
            raise ValueError(f"Invalid truncate_obs_side: {self.config.truncate_obs_side}")
        if self.config.enable_mtrl:
            next_obs = self.tokenizer.batch_decode(
                next_obs_ids,
                skip_special_tokens=True
            )
            processed_next_obs = []
            for i in range(len(next_obs)):
                if finishs[i] or dones[i]:
                    # do action is false
                    assert next_obs[i] == "", f"next_obs should be empty when finishs is True, but got {next_obs[i]}"
                    processed_next_obs.append("")
                elif valid_action[i]:
                    processed_next_obs.append(mtrl_sep.format(obs=next_obs[i]))
                else:
                    processed_next_obs.append(mtrl_sep.format(obs="Your action is not valid, please check the format and try again." + next_obs[i]))
            next_obs = processed_next_obs
            next_obs_ids = self.tokenizer(
                next_obs,
                padding='longest',
                return_tensors='pt',
                add_special_tokens=False,  # Prevents adding special tokens
            )['input_ids'].to(torch.int64)

        return next_obs_ids

    def _update_rolling_state(self, left_side, rollings, cur_responses: torch.Tensor,
                              next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""

        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_lens = new_attention_mask.sum(dim=1)
        effective_len = effective_lens.max()
        min_effective_len = effective_lens.min()
        # max_len = min(self.config.max_prompt_length, effective_len)
        max_len = min(self.config.max_prompt_length+self.config.max_response_length, effective_len)
        available_context_budget = max(0, self.config.max_prompt_length+self.config.max_response_length - min_effective_len)
        if self.config.max_action_length is not None and self.config.max_action_length > 0:
            available_context_budget = min(available_context_budget, self.config.max_action_length)
        if getattr(self.config, "rolling_with_prompt", False):
            # Added Zhiheng, if rolling_with_prompt is True, then we need to keep the system prompt
            if isinstance(left_side, dict):
                left_ids = left_side["input_ids"]
            else:
                left_ids = left_side.batch["input_ids"]

            left_len = left_ids.size(1)

            if left_len >= max_len:
                final_input_ids = left_ids[:, -max_len:]
            else:
                right_budget = max_len - left_len
                right_ids_full = new_input_ids[:, left_len:]
                right_ids = right_ids_full[:, -right_budget:] if right_budget < right_ids_full.size(1) else right_ids_full
                final_input_ids = torch.cat([left_ids, right_ids], dim=1)

            final_attention_mask = self.tensor_fn.create_attention_mask(final_input_ids)
            final_position_ids = self.tensor_fn.create_position_ids(final_attention_mask)

            new_rollings = DataProto.from_dict(
                {
                    "input_ids": final_input_ids,
                    "position_ids": final_position_ids,
                    "attention_mask": final_attention_mask,
                }
            )
        else: # By default keep the right side
            new_rollings = DataProto.from_dict(
                {
                    "input_ids": new_input_ids[:, -max_len:],
                    "position_ids": new_position_ids[:, -max_len:],
                    "attention_mask": new_attention_mask[:, -max_len:],
                }
            )
        new_rollings.non_tensor_batch = rollings.non_tensor_batch.copy()
        new_rollings.meta_info.update(rollings.meta_info)
        
        # update raw_prompt_ids, required for vllm inference
        ray_prompt_ids = []
        for i in range(new_rollings.batch['input_ids'].size(0)):
            non_pad_index = torch.nonzero(new_rollings.batch['input_ids'][i] != self.tokenizer.pad_token_id, as_tuple=False)[0][0]
            ray_prompt_ids.append(new_rollings.batch['input_ids'][i][non_pad_index:].tolist())
        new_rollings.non_tensor_batch['raw_prompt_ids'] = np.array(ray_prompt_ids, dtype=object)

        return new_rollings, available_context_budget

    def _info_masked_concatenate_with_padding(self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor = None,
        pad_to_left: bool = True
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        # move `response` and `info` tensor to the same device as `prompt`
        response = response.to(prompt.device)
        if info is not None:
            info = info.to(prompt.device)

        # set padding ids
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]

        # info: observations, need to be masked
        if info is not None:
            # for non-masked tensors, just append the observation
            tensors.append(info)

            # assemble the mask for the observation part
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)  # information mask
            # extend the mask for the observation part, to update masked tensors
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)

        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(
        self,
        right_side: Dict,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor = None
    ) -> Dict:
        """Update right side state."""

        # observation exists, perform concatenation and masked concatenation
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                next_obs_ids,
                pad_to_left=False
            )
        else:
            # no observation, only concatenate the response with generated response
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )

        effective_lens = self.tensor_fn.create_attention_mask(responses).sum(dim=1)
        effective_len = effective_lens.max()

        max_len = min(self.config.max_response_length, effective_len)

        overlong_dones = effective_lens >= self.config.max_response_length

        # return the updated responses along with its masked version
        if self.config.truncate_response_side == 'left':
            # it should be left most of the time.
            return {'responses': responses[:, :max_len],
                    'responses_with_info_mask': responses_with_info_mask[:, :max_len]}, overlong_dones
        elif self.config.truncate_response_side == 'right':
            return {'responses': responses[:, -max_len:],
                    'responses_with_info_mask': responses_with_info_mask[:, -max_len:]}, overlong_dones
        else:
            raise ValueError(
                f"Invalid truncate_response_side: {self.config.truncate_response_side}. Allowed options are 'left' or 'right'.")


    def run_llm_loop(self, gen_batch: DataProto) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        ori_meta_info = gen_batch.meta_info
        if isinstance(ori_meta_info['eos_token_id'], list):
            stop_token_ids = ori_meta_info['eos_token_id'] + self.additional_eos_token_ids
        else:
            stop_token_ids = [ori_meta_info['eos_token_id']] + self.additional_eos_token_ids
        gen_batch = self._preprocess_inputs(gen_batch)

        initial_input_ids = gen_batch.batch['input_ids'][:, -self.config.max_start_length:].clone()

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []],
                               'responses_with_info_mask': initial_input_ids[:, []]}

        turns_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool) # [bs*n]
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        traj_ids = gen_batch.non_tensor_batch['traj_ids']

        turns_stats_extra = {
            "action_lengths": [[] for _ in range(gen_batch.batch['input_ids'].shape[0])],
            "obs_lengths": [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        }

        agent_sampling_params = {
            "n": 1,  # already repeated by n times in _preprocess_inputs
            "stop": self.action_stop_tokens,  # stop when generated an end of action
            "include_stop_str_in_output": True,
            "detokenize": True,
            "stop_token_ids": stop_token_ids,
            # "allowed_token_ids": list(range(self.tokenizer.vocab_size)) # see vllm issue: # 1398
        }
        if self.config.max_action_length is not None and self.config.max_action_length > 0:
            agent_sampling_params['max_tokens'] = self.config.max_action_length

        if self.config.call_tool_first:
            # Added Zhiheng: Add initial observation to the prompt from server, use response=""
            do_actions = [True] * len(traj_ids)
            responses_str = [''] * len(traj_ids)
            responses_ids = torch.zeros((len(traj_ids), 1), dtype=torch.int64)
            active_uids = [traj_ids[i] for i in range(len(traj_ids)) if active_mask[i]]
            next_obs, dones, valid_action, finishs = self.interact_with_tool_server(
                active_uids, responses_str, do_actions, active_mask,
                extra_fields=rollings.non_tensor_batch.get('extra_info', None)
            )
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            # turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            next_obs_ids = self._process_next_obs(next_obs, dones, valid_action, finishs) # [active_size, obs_length]

            obs_idx = 0
            for i, active in enumerate(active_mask):
                if i >= len(turns_stats_extra["obs_lengths"]):
                    break
                if active:
                    obs_length = next_obs_ids[obs_idx].shape[0]
                    turns_stats_extra["obs_lengths"][i].append(int(obs_length))
                    obs_idx += 1
                else:
                    turns_stats_extra["obs_lengths"][i].append(0)

            rollings, available_context_budget = self._update_rolling_state(
                original_left_side,
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side, overlong_dones = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            agent_sampling_params['max_tokens'] = available_context_budget
            # print("Before overlong dones:", active_mask.sum().item())
            active_mask = active_mask * (~overlong_dones.to(active_mask.dtype).to(active_mask.device))
            # print("After overlong dones:", active_mask.sum().item())
            active_num_list.append(active_mask.sum().item())
            # End of Added Zhiheng

        # Main generation loop
        for step in range(self.config.max_turns+1):
            if not active_mask.sum():
                print("All trajectories are done.")
                break

            print(f"Action step {step}/{self.config.max_turns}")
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            ) # TODO: delete

            rollings_active = DataProto.from_dict(
                {k: v[active_mask] for k, v in rollings.batch.items()},
                {k: v[active_mask] for k, v in rollings.non_tensor_batch.items()},
                meta_info=ori_meta_info
            )
            if step == self.config.max_turns and self.config.force_finish_for_last_turn:
                # remove the action stop tokens in the last turn to force a finish
                agent_sampling_params.pop('stop')
            with self.actor_rollout_wg.rollout.update_sampling_params(**agent_sampling_params):
                gen_output = self.actor_rollout_wg.rollout.generate_sequences(rollings_active) # [active_size, response_length]

            responses_ids, responses_str, do_actions = self._postprocess_responses(gen_output.batch['responses'], step) # [active_size, ...]
            responses_ids, _ = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask) # [bs*n, response_length]

            print(f"Number of active trajectories: {active_mask.sum().item()}")
            print(f"Length of responses: {responses_ids.shape[1]}")

            idx = 0
            for i, active in enumerate(active_mask):
                if active:
                    action_length = len(self.tokenizer.encode(responses_str[idx], add_special_tokens=False))
                    turns_stats_extra["action_lengths"][i].append(action_length)
                    idx += 1
                else:
                    turns_stats_extra["action_lengths"][i].append(0)

            # Execute in environment and process observations
            active_uids = [traj_ids[i] for i in range(len(traj_ids)) if active_mask[i]]
            next_obs, dones, valid_action, finishs = self.interact_with_tool_server(
                active_uids, responses_str, do_actions, active_mask,
                extra_fields=rollings_active.non_tensor_batch.get('extra_info', None),
                is_last_step=(step == self.config.max_turns)
            )

            # # for debug
            # with open(f"temp-{step}.json", 'w') as f:
            #     json.dump([{
            #         'prompt': self.tokenizer.decode(rollings_active.batch['input_ids'][i], skip_special_tokens=False),
            #         'response': resp,
            #         'do_action': do_action,
            #         'traj_id': traj_id,
            #         'next_obs': next_obs[i],
            #         'done': done,
            #         'valid_action': valid_action[i],
            #     } for i, (resp, do_action, traj_id, done) in enumerate(zip(responses_str, do_actions, active_uids, dones))], f, indent=4)
            #     print(f"saved responses to temp-{step}.json")

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs, dones, valid_action, finishs) # [active_size, obs_length]

            obs_idx = 0
            for i, active in enumerate(active_mask):
                if i >= len(turns_stats_extra["obs_lengths"]):
                    break
                if active:
                    obs_length = next_obs_ids[obs_idx].shape[0]
                    turns_stats_extra["obs_lengths"][i].append(int(obs_length))
                    obs_idx += 1
                else:
                    turns_stats_extra["obs_lengths"][i].append(0)

            # Update states
            rollings, available_context_budget = self._update_rolling_state(
                original_left_side,
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side, overlong_dones = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            agent_sampling_params['max_tokens'] = available_context_budget
            # print("Before overlong dones:", active_mask.sum().item())
            active_mask = active_mask * (~overlong_dones.to(active_mask.dtype).to(active_mask.device))
            # print("After overlong dones:", active_mask.sum().item())
            active_num_list.append(active_mask.sum().item())

        non_tensors = {
            'traj_ids': traj_ids.tolist(),
            'turns_stats': turns_stats.tolist(),
            'valid_action_stats': valid_action_stats.tolist(),
            'active_mask': active_mask.tolist(),
            'action_lengths': turns_stats_extra["action_lengths"],
            'obs_lengths': turns_stats_extra["obs_lengths"],
        }

        print("ACTIVE_TRAJ_NUM:", active_num_list)

        results = self._compose_final_output(original_left_side, original_right_side, non_tensors, ori_meta_info)
        return results

    def _compose_final_output(
        self,
        left_side: Dict,
        right_side: Dict,
        non_tensors: Dict,
        meta_info: Dict
    ) -> Tuple[Dict, Dict]:
        """
        Compose the final output of the rollout by merging prompt and response
        components, padding sequences as needed, and ensuring all turn-level
        non-tensor fields are aligned in shape for safe concatenation across samples.
        """
        # ---------- 1. Pad turn-level lists to the same length ----------
        pad_len = self.config.max_turns + 2  # buffer to avoid mismatch

        def _pad(seq_list, fill_value=0):
            """
            Pad or truncate a list to match pad_len.
            This is used for per-turn statistics like action_lengths or obs_lengths.
            """
            if len(seq_list) < pad_len:
                seq_list += [fill_value] * (pad_len - len(seq_list))
            else:
                seq_list[:] = seq_list[:pad_len]
            return seq_list

        if "action_lengths" in non_tensors:
            non_tensors["action_lengths"] = [
                _pad(traj, 0) for traj in non_tensors["action_lengths"]
            ]
        if "obs_lengths" in non_tensors:
            non_tensors["obs_lengths"] = [
                _pad(traj, 0) for traj in non_tensors["obs_lengths"]
            ]

        # ---------- 2. Build final tensor fields ----------
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids'] # [bs*n, prompt_length]

        # padding responses length to max_response_length
        if final_output['responses'].shape[1] < self.config.max_response_length:
            final_output['responses'] = self.tensor_fn.pad_tensor(
                final_output['responses'],
                max_length=self.config.max_response_length,
                padding_side='right'
            ) # [bs*n, max_response_length]

        # padding response_with_info_mask length to max_response_length
        if final_output['responses_with_info_mask'].shape[1] < self.config.max_response_length:
            final_output['responses_with_info_mask'] = self.tensor_fn.pad_tensor(
                final_output['responses_with_info_mask'],
                max_length=self.config.max_response_length,
                padding_side='right'
            ) # [bs*n, max_response_length]

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            final_output['responses']
        ], dim=1) # [bs*n, prompt_length + max_response_length]

        # Create attention mask
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1) # [bs*n, prompt_length + max_response_length]

        # Create observation mask
        if self.config.mask_observations:
            final_output['info_mask'] = torch.cat([
                self.tensor_fn.create_attention_mask(left_side['input_ids']),
                self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
            ], dim=1) # [bs*n, prompt_length + max_response_length]
        else:
            final_output['info_mask'] = final_output['attention_mask']

        # Create position ids
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        ) # [bs*n, prompt_length + max_response_length]

        # ---------- 3. Create and return DataProto ----------
        final_output = DataProto.from_dict(final_output, non_tensors=non_tensors)
        final_output.meta_info.update(meta_info)

        return final_output

    def send_batch_requests(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send batch requests to the tool server.
        Args:
            batch_data: Batch data to send
        Returns:
            response: Response from the tool server
        """
        safe_payload = sanitize_request(batch_data)
        response = requests.post(self.config.tool_server_url, json=safe_payload)
        if response.status_code != 200:
            print("#" * 100)
            # print(safe_payload)
            print("URL:", self.config.tool_server_url)
            print("Length of trajectory ids:", len(safe_payload['trajectory_ids']))
            print("Length of actions:", len(safe_payload['actions']))
            print("Length of finish:", len(safe_payload['finish']))
            print("Length of extra fields:", len(safe_payload.get('extra_fields', [])))
            print("$" * 100)
            print(response)
            print("#" * 100)
            with open("error_data.json", 'w') as f:
                json.dump(batch_data, f, indent=4)
            try:
                # Try to decode as utf-8 for error message
                error_text = response.text
                print(f"Error: {response.status_code}, {error_text}")
            except UnicodeDecodeError:
                # If decoding fails, show raw content and encoding
                print(f"Error: {response.status_code}, Binary response, encoding: {response.encoding}")
                print(f"Raw content (first 100 bytes): {response.content[:100]}")
            raise ValueError(f"Error: {response.status_code}, Response could not be decoded as UTF-8")
        
        try:
            return response.json()
        except ValueError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response content type: {response.headers.get('Content-Type')}")
            print(f"First 100 chars of response: {response.text[:100]}")
            raise
        
        # if response.status_code != 200:
        #     print(f"Error: {response.status_code}, {response.text}")
        #     with open("error_data.json", 'w') as f:
        #         json.dump(batch_data, f, indent=4)
        #     raise ValueError(f"Error: {response.status_code}, {response.text}")
        # return response.json()

    def interact_with_tool_server(
        self,
        active_uids:List[str],
        responses: List[str],
        do_actions:List[bool],
        active_mask=None,
        extra_fields=None,
        is_last_step=False,
    ) -> List[str]:
        """
        Call tool server for queries.
        Args:
            batch: batch of data
            resposnes: responses from the model
            pad_token: pad token
            active_mask: active mask
        Returns:
            observations: observations from the tool server. None if the the query do not need to do any action.
            dones: dones
            valid_actions: valid actions
        """
        finishs = [not do_action for do_action in do_actions]
        batch_data = {
            "trajectory_ids": active_uids,
            "actions": responses,
            "finish": finishs, # if do_action is False, then it is a finish action, finishing the trajectory,
            "is_last_step": [is_last_step] * len(finishs)
        }
        if extra_fields is not None:
            # ef = list(extra_fields)
            # TODO: Figure out why this is needed, temporarily disabled
            # if len(ef) != len(active_mask):
            #     ef = ef + [{}] * (len(active_mask) - len(ef))
            # active_extra_fields = [ef[i] for i in range(len(ef)) if active_mask[i]]
            batch_data['extra_fields'] = extra_fields
        print(f" - Number of finished responses: {len([x for x in do_actions if not x])} / {len(do_actions)}")
        response = self.send_batch_requests(batch_data)
        active_observations = response['observations']
        active_dones = [int(x) for x in response['dones']]
        active_valid_actions = [int(x) for x in response['valids']]

        # print("Received observations from tool server. Samples:", len(active_observations))
        # print(f" - Number of valid actions (exclusing finish action): {len([x for x in active_valid_actions if x])} / {len(active_valid_actions)}")
        # print(f" - Number of dones: {len([x for x in active_dones if x])} / {len(active_dones)}")
        # print("Example observations:")
        # non_empty_observations = [obs for obs in active_observations if obs]
        # if len(non_empty_observations) > 0:
        #     print(f"{non_empty_observations[0]}")
        # else:
        #     print("No non-empty observations.")

        next_obs, dones, valid_action, _finishs = [], [], [], []
        for i, active in enumerate(active_mask):
            if active:
                next_obs.append(active_observations.pop(0))
                dones.append(active_dones.pop(0)) # whether the trajectory is finished for eos or considered done by the remote server
                valid_action.append(active_valid_actions.pop(0))
                _finishs.append(finishs.pop(0)) # whether the trajectory is finished for eos
            else:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                _finishs.append(1)

        assert len(active_observations) == 0
        return next_obs, dones, valid_action, _finishs
