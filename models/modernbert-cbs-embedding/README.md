---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:13697
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'The weather is lovely today.',
    "It's so sunny outside!",
    'He drove to the stadium.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 13,697 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0         | sentence_1         | label                                                          |
  |:--------|:-------------------|:-------------------|:---------------------------------------------------------------|
  | type    | dict               | dict               | float                                                          |
  | details | <ul><li></li></ul> | <ul><li></li></ul> | <ul><li>min: 0.0</li><li>mean: 0.98</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>{'page_name': 'Supply Chain Management', 'section_title': 'Potentially Qualifying Bachelor Degrees(Can Supplement to Qualify)', 'section_type': 'Potentially Qualifying Bachelor Degrees(Can Supplement to Qualify)', 'text': 'page_name: Supply Chain Managementtext: Your obligatory courses do not fully cover the academic requirements. You will therefore need to either choose relevant electives or take supplementary qualifying courses to fulfil the requirements.You can use the self-assessment form (further down the page) to give an overview of any courses you believe fulfil the missing academic entry requirements from the advance assessment. All obligatory courses have been taken into consideration.The academic requirements must be covered with academic and research-based courses done at bachelor level. All obligatory courses have been taken into consideration.The academic requirements must be covered with academic and research-based courses done at bachelor level.', 'url': 'https://www.cbs.dk/en/study/graduate/supply-chain-management/entry-requirements'}</code> | <code>{'page_name': 'Supply Chain Management', 'section_title': 'Qualifying Bachelor Degrees (Part 2)', 'section_type': 'Qualifying Bachelor Degrees (Part 2)', 'text': 'page_name: Supply Chain Managementtext: BI Handelshøyskolen - Bachelor of Data Science When applying, please specify the exact name of the bachelor programme you are enrolled in, if this is not written in your grade transcript.', 'url': 'https://www.cbs.dk/en/study/graduate/supply-chain-management/entry-requirements'}</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | <code>1.0</code> |
  | <code>{'page_name': 'Diversity and Change Management', 'section_title': 'ADMISSION PROCESS AND DEADLINES', 'section_type': 'ADMISSION PROCESS AND DEADLINES', 'text': 'page_name: Diversity and Change Managementtext: Learn more about the admission process and find answers to questions such as:', 'url': 'https://www.cbs.dk/en/study/graduate/diversity-and-change-management/entry-requirements'}</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | <code>{'page_name': 'Diversity and Change Management', 'section_title': '1. English language requirement', 'section_type': '1. English language requirement', 'text': 'page_name: Diversity and Change Managementtext: The programme requires English at Danish level A. The English requirements are very strict, which means that only the qualifications listed on the English level A page are accepted as alternative fulfilment.The English language requirement must be fulfilled by the application deadline.', 'url': 'https://www.cbs.dk/en/study/graduate/diversity-and-change-management/entry-requirements'}</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | <code>1.0</code> |
  | <code>{'page_name': 'Finance and Strategic Management', 'section_title': 'Advance Assessments', 'section_type': 'Advance Assessments', 'text': 'page_name: Finance and Strategic Managementtext: The following advance assessments are only valid for: What is included in the advance assessment?The advance assessments are based on the obligatory courses as listed in the programme regulations.If you have obligatory courses that have been credit transferred from elsewhere, we will not use the advance assessment and will instead assess your courses using the available documentation.If you have electives, we will assess these in addition to the obligatory courses.CategoriesThe advance assessments are grouped into 3 different categories: Your bachelor degree may be listed in different categories for different enrolment years.Please note: Fulfilment of the entry requirements does not guarantee admission. Read more under ‘Selection Criteria’. Bachelor degrees from CBS', 'url': 'https://www.cbs.dk/en/study/graduate/finance-and-strategic-management/entry-requirements'}</code>   | <code>{'page_name': 'Finance and Strategic Management', 'section_title': 'Potentially Qualifying Bachelor Degrees(Can Supplement to Qualify) (Part 2)', 'section_type': 'Potentially Qualifying Bachelor Degrees(Can Supplement to Qualify) (Part 2)', 'text': 'page_name: Finance and Strategic Managementtext: All obligatory courses have been taken into consideration.The academic requirements must be covered with academic and research-based courses done at bachelor level.*The Organisation requirement is fulfilled if "Organization Theory" is chosen as one of the third year major compulsory courses and passed at Bocconi University (not on exchange).**The Marketing requirement is fulfilled if "Marketing" is chosen as one of the third year major compulsory courses and passed at Bocconi University (not on exchange). University of Bologna – Bachelor in Economics and Finance (taught in English) NB: In case of a credit transfer of obligatory courses, we reserve the right not to use the advance assessment. All obligatory courses have been taken into consideration.The academic requirements must be covered with academic and research-based courses done at bachelor level.', 'url': 'https://www.cbs.dk/en/study/graduate/finance-and-strategic-management/entry-requirements'}</code> | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 5
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.5834 | 500  | 0.0047        |
| 1.1669 | 1000 | 0.0021        |
| 1.7503 | 1500 | 0.0012        |
| 2.3337 | 2000 | 0.0012        |
| 2.9172 | 2500 | 0.0009        |
| 3.5006 | 3000 | 0.0007        |
| 4.0840 | 3500 | 0.0006        |
| 4.6674 | 4000 | 0.0006        |


### Framework Versions
- Python: 3.11.12
- Sentence Transformers: 3.4.1
- Transformers: 4.51.3
- PyTorch: 2.6.0+cu124
- Accelerate: 1.6.0
- Datasets: 3.5.1
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->