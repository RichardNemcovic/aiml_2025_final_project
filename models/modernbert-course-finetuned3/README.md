---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:38096
- loss:ContrastiveLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: What is the ECTS value of the course 2024/2025BA-BPOLV3001U  BSc
    IBP Internship, 15 ECTS?
  sentences:
  - 2024/2025DIP-DIBUO1005U  Practical Seminar is a BachelorGraduate DiplomaFull Degree
    MasterMasterPart Time Master level course.
  - 'Bachelor level

    Examination

    Risk

    Management and Corporate Finance:

    Exam

    ECTS

    7,5

    Examination form

    Written sit-in exam on CBS''

    computers

    Individual or group exam

    Individual exam

    Assignment type

    Written assignment

    Duration

    4 hours

    Grading scale

    7-point grading scale

    Examiner(s)

    One internal examiner

    Exam period

    Autumn

    Aids

    Limited aids, see the list below:

    The student is allowed to bring

    USB key for uploading of notes, books and compendiums in a

    non-executable format (no applications, application fragments, IT

    tools etc.)

    Any calculator

    In Paper format: Books (including translation dictionaries),

    compendiums and notes

    The student will have access to

    Access to Canvas

    Access to the personal drive (S-drive) on CBS´ network

    Advanced IT application package

    Read more here :

    Exam aids and IT application

    packages

    Make-up exam/re-exam

    Same examination form as the ordinary exam

    The number of registered candidates for the make-up

    examination/re-take examination may warrant that it most

    appropriately be held as an oral examination. The programme office

    will inform the students if the make-up examination/re-take

    examination instead is held as an oral examination including a

    second examiner or external

    examiner. Course content, structure and pedagogical

    approach

    The course builds on the law of one price as the underlying

    theoretical principle and discusses how financial frictions affects

    decision-making. It is a fundamental finance course, which

    complements introductory bachelor level corporate finance

    courses. The first part of the course provides the students with an

    understanding of corporate finance. The core elements covered in

    this part of the course are the valuation of investment projects,

    corporate finance polices, payout policy, and capital structure

    choice. The second part of the course investigates corporate risk

    management and covers financial options, real options, hedging of

    corporate risk, and international risk management. Research-based teaching

    CBS’ programmes and teaching are research-based.'
  - 2024/2025BA-BPOLV3001U  BSc IBP Internship, 15 ECTS is a 15 ECTS ECTS course.
- source_sentence: Is 2024/2025BA-BBLCO2102U  Business Study Report a mandatory or
    elective course?
  sentences:
  - 2025/2026KAN-CPOLO2402U  Evidence Based Public Policy is a Mandatory (also offered
    as elective) course.
  - 2024/2025BA-BBLCO2102U  Business Study Report is a MandatoryMandatory (also offered
    as elective)Elective course.
  - '2024/2025BA-BHAAV5009U  Strategic Management of Innovation and

    Technology is a 7.5 ECTS ECTS course.'
- source_sentence: What level is the course 2025/2026KAN-CIHCO2007U  Managerial Statistics
    for Innovation?
  sentences:
  - '2024/2025

    BA-BSEMV2301U  Strategic use of Intellectual Property in

    Service industries

    English Title

    Strategic use of Intellectual

    Property in Service industries

    Course information

    Language

    English

    Course ECTS

    7.5 ECTS

    Type

    Elective

    Level

    Bachelor

    Duration

    One Quarter

    Start time of the course

    First Quarter

    Timetable

    Course schedule will be posted at

    calendar.cbs.dk

    Max. participants

    100

    Study board

    Study Board for BSc in Service

    Management

    Course

    coordinator

    Vishv Priya Kohli - Department of Business Humanities and Law

    (BHL)

    Main academic

    disciplines

    Business Law

    Sociology

    Teaching

    methods

    Blended learning

    Last updated on

    18-01-2024

    Relevant links

    Programme Regulations

    Rules and

    regulations for exams at CBS

    Learning objectives

    The aim of the course is to enable students to

    Analyse, articulate and account for the creation and

    preservation of Intellectual Property in the service

    industries

    Select and apply IP rules relevant to business processes in the

    service industries

    Critically analyze business problems in specific contexts in

    order to constructively manage the IP in the creative

    industries

    Understand the challenges, assess the implications, and apply

    IP law in service management

    Examination

    Strategic Use

    of Intellectual Property in Service Industries:

    Exam

    ECTS

    7,5

    Examination form

    Home assignment - written product

    Individual or group exam

    Individual exam

    Size of written product

    Max. 10 pages

    Assignment type

    Case based assignment

    Release of assignment

    The Assignment is released in Digital Exam (DE)

    at exam start

    Duration

    72 hours to prepare

    Grading scale

    7-point grading scale

    Examiner(s)

    One internal examiner

    Exam period

    Winter

    Make-up exam/re-exam

    Same examination form as the ordinary

    exam

    Course content, structure and pedagogical

    approach

    The course is designed to give practical tools that can be

    applied by students – the future managers, in the service

    industries to develop, protect and maintain the Intellectual

    Property (IP) assets.'
  - 2025/2026KAN-CIHCO2007U  Managerial Statistics for Innovation is a BachelorGraduate
    DiplomaFull Degree MasterMasterPart Time Master level course.
  - 2025/2026KAN-CSIEO2006U  The Art of Innovation is a 7.5 ECTS ECTS course.
- source_sentence: What level is the course 2024/2025DIP-DHDVV2001U  Behavioural Finance?
  sentences:
  - '2024/2025KAN-CCMVA8005U  Sustainable Business in Northern

    Europe is taught in English.'
  - 2024/2025BA-BHAAV2260U  Entrepreneurial Strategy is a Elective course.
  - 2024/2025DIP-DHDVV2001U  Behavioural Finance is a BachelorGraduate DiplomaFull
    Degree MasterMasterPart Time Master level course.
- source_sentence: What level is the course 2024/2025KAN-CEADO1002U  Asset Pricing?
  sentences:
  - '2024/2025KAN-CBIOO1007U  Finance and Accounting in

    Bio-Business is taught in English.'
  - 2024/2025BA-BDMAO1001U  Managing Innovation in Organizations is taught in English.
  - 2024/2025KAN-CEADO1002U  Asset Pricing is a BachelorGraduate DiplomaFull Degree
    MasterMasterPart Time Master level course.
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
    'What level is the course 2024/2025KAN-CEADO1002U\xa0\xa0Asset Pricing?',
    '2024/2025KAN-CEADO1002U\xa0\xa0Asset Pricing is a BachelorGraduate DiplomaFull Degree MasterMasterPart Time Master level course.',
    '2024/2025BA-BDMAO1001U\xa0\xa0Managing Innovation in Organizations is taught in English.',
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

* Size: 38,096 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                           | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                               | float                                                          |
  | details | <ul><li>min: 21 tokens</li><li>mean: 27.62 tokens</li><li>max: 41 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 109.67 tokens</li><li>max: 437 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.61</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                 | sentence_1                                                                                                                                                  | label            |
  |:-----------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What is the ECTS value of the course 2025/2026BA-BINBO1337U  International Business Strategy?</code> | <code>2025/2026BA-BINBO1337U  International Business Strategy is a 7.5 ECTS ECTS course.</code>                                                             | <code>1.0</code> |
  | <code>Is 2024/2025BA-BHAAV2260U  Entrepreneurial Strategy a mandatory or elective course?</code>           | <code>2024/2025BA-BHAAV2260U  Entrepreneurial Strategy is a Elective course.</code>                                                                         | <code>1.0</code> |
  | <code>What level is the course 2024/2025KAN-CCMVV2453U  International Marketing and Sales?</code>          | <code>2024/2025KAN-CCMVV2453U  International Marketing and Sales is a BachelorGraduate DiplomaFull Degree MasterMasterPart Time Master level course.</code> | <code>0.0</code> |
* Loss: [<code>ContrastiveLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#contrastiveloss) with these parameters:
  ```json
  {
      "distance_metric": "SiameseDistanceMetric.COSINE_DISTANCE",
      "margin": 0.5,
      "size_average": true
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 4
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
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
- `num_train_epochs`: 4
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
- `fp16`: True
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
| 0.4198 | 500  | 0.0057        |
| 0.8396 | 1000 | 0.0024        |
| 1.2594 | 1500 | 0.0019        |
| 1.6793 | 2000 | 0.0016        |
| 2.0991 | 2500 | 0.0014        |
| 2.5189 | 3000 | 0.0013        |
| 2.9387 | 3500 | 0.0012        |
| 3.3585 | 4000 | 0.001         |
| 3.7783 | 4500 | 0.001         |


### Framework Versions
- Python: 3.10.16
- Sentence Transformers: 4.1.0
- Transformers: 4.51.3
- PyTorch: 2.5.1+cu121
- Accelerate: 1.6.0
- Datasets: 3.6.0
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

#### ContrastiveLoss
```bibtex
@inproceedings{hadsell2006dimensionality,
    author={Hadsell, R. and Chopra, S. and LeCun, Y.},
    booktitle={2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06)},
    title={Dimensionality Reduction by Learning an Invariant Mapping},
    year={2006},
    volume={2},
    number={},
    pages={1735-1742},
    doi={10.1109/CVPR.2006.100}
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