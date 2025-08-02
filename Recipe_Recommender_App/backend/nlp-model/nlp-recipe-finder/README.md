---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:463274
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: time-to-make main-ingredient preparation occasion vegetables 1-day-or-more
    easy refrigerator beginner-cook dinner-party holiday-event summer vegetarian stove-top
    dietary seasonal comfort-food onions peppers taste-mood equipment presentation
    served-cold green beans canned corn niblets frozen peas green bell pepper red
    onion sugar white vinegar vegetable oil salt pepper garlic powder
  sentences:
  - 60-minutes-or-less time-to-make course main-ingredient cuisine preparation occasion
    main-dish vegetables asian chinese easy diabetic dinner-party dietary carrots
    peppers presentation served-hot boneless beef top sirloin steaks garlic cloves
    olive oil carrots celery bell peppers water chestnuts bamboo shoots pineapple
    oyster sauce garlic sauce cayenne pepper salt pepper cinnamon lemon zest spinach
    leaves soy sauce
  - celebrity weeknight time-to-make course main-ingredient cuisine preparation occasion
    north-american main-dish american easy beginner-cook potluck fall holiday-event
    spring summer winter chili dietary spicy low-sodium seasonal low-calorie comfort-food
    low-carb low-in-something novelty taste-mood to-go jalapenos onions garlic clove
    tomato sauce salt and pepper lean ground beef cumin seed oregano cold water
  - 30-minutes-or-less time-to-make course main-ingredient cuisine preparation occasion
    for-1-or-2 omelets-and-frittatas breakfast main-dish eggs-dairy greek european
    cheese eggs stove-top one-dish-meal meat brunch equipment number-of-servings butter
    eggs milk ground cayenne pepper white pepper tabasco sauce salt feta cheese onion
    gyro meat diced tomato cucumbers
- source_sentence: ham time-to-make course main-ingredient preparation main-dish pork
    oven easy meat equipment 3-steps-or-less 4-hours-or-less ham cranberry sauce orange
    juice a.1. original sauce brown sugar vegetable oil mustard
  sentences:
  - 60-minutes-or-less time-to-make course main-ingredient preparation side-dishes
    grains pasta-rice-and-grains quinoa milk vegetable broth butter onion sweet red
    pepper salt and pepper frozen mixed vegetables all-purpose flour
  - 60-minutes-or-less time-to-make course main-ingredient cuisine preparation casseroles
    main-dish beans beef pasta pork oven european italian dietary one-dish-meal meat
    pork-sausage pasta-rice-and-grains equipment hot italian sausage ground beef onion
    garlic dried oregano dried thyme italian plum tomatoes tomato paste cayenne pepper
    kidney beans salt and pepper mostaccioli pasta parmesan cheese fresh italian parsley
    fontina
  - 60-minutes-or-less time-to-make course preparation occasion for-large-groups desserts
    oven easy kid-friendly cakes dietary gifts comfort-food taste-mood equipment number-of-servings
    butter granulated sugar eggs vanilla cocoa all-purpose flour baking soda baking
    powder salt beer sauerkraut
- source_sentence: time-to-make course main-ingredient cuisine preparation occasion
    healthy appetizers lunch eggs-dairy greek european dinner-party vegetarian cheese
    dietary low-sodium low-in-something brunch 4-hours-or-less light butter white
    onions sugar balsamic vinegar honey fresh thyme phyllo pastry sheets butter-flavored
    cooking spray goat cheese
  sentences:
  - 15-minutes-or-less time-to-make course preparation salads salad-dressings garlic
    cloves extra virgin olive oil salt & freshly ground black pepper balsamic vinegar
    red wine vinegar dark brown sugar
  - lactose 15-minutes-or-less time-to-make course main-ingredient cuisine preparation
    occasion north-american pancakes-and-waffles breads breakfast rice easy grains
    dietary gluten-free inexpensive egg-free free-of-something pasta-rice-and-grains
    white-rice brunch 3-steps-or-less from-scratch tapioca flour cinnamon nutmeg cooking
    oil baking powder water brown sugar rice flour
  - 60-minutes-or-less time-to-make course main-ingredient preparation 5-ingredients-or-less
    very-low-carbs appetizers poultry easy beginner-cook kid-friendly chicken dietary
    high-protein low-carb high-in-something low-in-something meat 3-steps-or-less
    chicken drummettes soy sauce apricot jam water garlic clove
- source_sentence: 60-minutes-or-less time-to-make course main-ingredient preparation
    5-ingredients-or-less casseroles side-dishes vegetables oven easy equipment frozen
    broccoli carrots cauliflower mix condensed cream of mushroom soup cream cheese
    with vegetables seasoned croutons
  sentences:
  - time-to-make main-ingredient cuisine preparation occasion north-american poultry
    american mexican southwestern-united-states central-american dinner-party chicken
    crock-pot-slow-cooker food-processor-blender dietary spicy meat whole-chicken
    taste-mood equipment small-appliance blanched almond cinnamon stick coriander
    seed anise seed sesame seeds whole cloves dried pasilla peppers dried new mexico
    chiles guajillo chilies chicken stock roma tomatoes mexican chocolate garlic cloves
    sweet white onion raisins salt chicken
  - 30-minutes-or-less time-to-make course preparation hand-formed-cookies desserts
    cookies-and-brownies flour baking soda salt sugar milk brown sugar shortening
    creamy peanut butter egg vanilla hershey chocolate kisses
  - 60-minutes-or-less time-to-make course main-ingredient preparation 5-ingredients-or-less
    casseroles side-dishes vegetables oven easy equipment frozen broccoli carrots
    cauliflower mix condensed cream of mushroom soup cream cheese with vegetables
    seasoned croutons
- source_sentence: bacon 60-minutes-or-less time-to-make course main-ingredient cuisine
    preparation occasion north-american side-dishes pork potatoes vegetables american
    oven roast fall holiday-event dietary thanksgiving seasonal comfort-food northeastern-united-states
    meat taste-mood savory equipment presentation served-hot fingerling potatoes bacon
    onions garlic clove fresh sage leaves coarse salt
  sentences:
  - 60-minutes-or-less time-to-make course main-ingredient cuisine preparation south-african
    desserts eggs-dairy african inexpensive milk cake crumbs mashed banana sugar salt
    lemon juice vanilla eggs
  - time-to-make course main-ingredient preparation healthy very-low-carbs main-dish
    pork dietary low-sodium high-protein low-carb high-in-something low-in-something
    meat 4-hours-or-less pork roast olive oil salt and pepper garlic cloves onion
    water thyme sage dry red wine flour
  - bacon 60-minutes-or-less time-to-make course main-ingredient cuisine preparation
    occasion north-american side-dishes pork potatoes vegetables american oven roast
    fall holiday-event dietary thanksgiving seasonal comfort-food northeastern-united-states
    meat taste-mood savory equipment presentation served-hot fingerling potatoes bacon
    onions garlic clove fresh sage leaves coarse salt
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
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
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
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
    'bacon 60-minutes-or-less time-to-make course main-ingredient cuisine preparation occasion north-american side-dishes pork potatoes vegetables american oven roast fall holiday-event dietary thanksgiving seasonal comfort-food northeastern-united-states meat taste-mood savory equipment presentation served-hot fingerling potatoes bacon onions garlic clove fresh sage leaves coarse salt',
    'bacon 60-minutes-or-less time-to-make course main-ingredient cuisine preparation occasion north-american side-dishes pork potatoes vegetables american oven roast fall holiday-event dietary thanksgiving seasonal comfort-food northeastern-united-states meat taste-mood savory equipment presentation served-hot fingerling potatoes bacon onions garlic clove fresh sage leaves coarse salt',
    '60-minutes-or-less time-to-make course main-ingredient cuisine preparation south-african desserts eggs-dairy african inexpensive milk cake crumbs mashed banana sugar salt lemon juice vanilla eggs',
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

* Size: 463,274 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                          | label                                                          |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              | float                                                          |
  | details | <ul><li>min: 18 tokens</li><li>mean: 74.79 tokens</li><li>max: 148 tokens</li></ul> | <ul><li>min: 18 tokens</li><li>mean: 73.49 tokens</li><li>max: 170 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.52</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | sentence_1                                                                                                                                                                                                                                                                                                                                               | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>30-minutes-or-less time-to-make course main-ingredient cuisine preparation occasion north-american healthy 5-ingredients-or-less soups-stews beans vegetables american southwestern-united-states tex-mex easy beginner-cook fall low-fat vegan vegetarian winter dietary low-cholesterol seasonal low-saturated-fat low-calorie comfort-food inexpensive black-beans healthy-2 low-in-something taste-mood 3-steps-or-less vegetable broth refried beans black beans frozen corn rotel tomatoes & chilies</code> | <code>time-to-make course preparation occasion for-large-groups desserts cakes gifts copycat novelty number-of-servings 4-hours-or-less cake flour baking powder salt semisweet chocolate unsalted butter granulated sugar vanilla eggs orange liqueur lemons, zest of</code>                                                                            | <code>0.0</code> |
  | <code>60-minutes-or-less time-to-make course cuisine preparation for-large-groups desserts european cakes number-of-servings seedless raisins walnuts baking soda boiling water all-purpose flour cinnamon salt butter sugar eggs egg yolks lemon juice vanilla extract cream cheese confectioners' sugar</code>                                                                                                                                                                                                        | <code>60-minutes-or-less time-to-make course cuisine preparation for-large-groups desserts european cakes number-of-servings seedless raisins walnuts baking soda boiling water all-purpose flour cinnamon salt butter sugar eggs egg yolks lemon juice vanilla extract cream cheese confectioners' sugar</code>                                         | <code>1.0</code> |
  | <code>60-minutes-or-less time-to-make course main-ingredient preparation main-dish soups-stews poultry easy beginner-cook turkey meat butter all-purpose flour milk salt chicken broth herbes de provence bay leaves celery seed garlic powder lemon pepper ground black pepper cream cheese water parmesan cheese wide egg noodles cooked turkey</code>                                                                                                                                                                | <code>60-minutes-or-less time-to-make course main-ingredient preparation main-dish soups-stews poultry easy beginner-cook turkey meat butter all-purpose flour milk salt chicken broth herbes de provence bay leaves celery seed garlic powder lemon pepper ground black pepper cream cheese water parmesan cheese wide egg noodles cooked turkey</code> | <code>1.0</code> |
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
- `num_train_epochs`: 3
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
- `hub_revision`: None
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
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0173 | 500   | 0.0158        |
| 0.0345 | 1000  | 0.0075        |
| 0.0518 | 1500  | 0.0068        |
| 0.0691 | 2000  | 0.0063        |
| 0.0863 | 2500  | 0.006         |
| 0.1036 | 3000  | 0.006         |
| 0.1209 | 3500  | 0.0056        |
| 0.1381 | 4000  | 0.0055        |
| 0.1554 | 4500  | 0.0053        |
| 0.1727 | 5000  | 0.0052        |
| 0.1899 | 5500  | 0.005         |
| 0.2072 | 6000  | 0.0046        |
| 0.2245 | 6500  | 0.0049        |
| 0.2418 | 7000  | 0.0046        |
| 0.2590 | 7500  | 0.0047        |
| 0.2763 | 8000  | 0.0049        |
| 0.2936 | 8500  | 0.0045        |
| 0.3108 | 9000  | 0.0045        |
| 0.3281 | 9500  | 0.0047        |
| 0.3454 | 10000 | 0.0044        |
| 0.3626 | 10500 | 0.0046        |
| 0.3799 | 11000 | 0.0041        |
| 0.3972 | 11500 | 0.0043        |
| 0.4144 | 12000 | 0.0044        |
| 0.4317 | 12500 | 0.0043        |
| 0.4490 | 13000 | 0.0039        |
| 0.4662 | 13500 | 0.0042        |
| 0.4835 | 14000 | 0.0041        |
| 0.5008 | 14500 | 0.0043        |
| 0.5180 | 15000 | 0.0042        |
| 0.5353 | 15500 | 0.0038        |
| 0.5526 | 16000 | 0.004         |
| 0.5698 | 16500 | 0.0037        |
| 0.5871 | 17000 | 0.0038        |
| 0.6044 | 17500 | 0.0041        |
| 0.6217 | 18000 | 0.0038        |
| 0.6389 | 18500 | 0.004         |
| 0.6562 | 19000 | 0.0037        |
| 0.6735 | 19500 | 0.0035        |
| 0.6907 | 20000 | 0.0036        |
| 0.7080 | 20500 | 0.0035        |
| 0.7253 | 21000 | 0.0037        |
| 0.7425 | 21500 | 0.0038        |
| 0.7598 | 22000 | 0.0037        |
| 0.7771 | 22500 | 0.0037        |
| 0.7943 | 23000 | 0.0035        |
| 0.8116 | 23500 | 0.0034        |
| 0.8289 | 24000 | 0.0034        |
| 0.8461 | 24500 | 0.0037        |
| 0.8634 | 25000 | 0.0033        |
| 0.8807 | 25500 | 0.0032        |
| 0.8979 | 26000 | 0.0033        |
| 0.9152 | 26500 | 0.0036        |
| 0.9325 | 27000 | 0.0035        |
| 0.9497 | 27500 | 0.0032        |
| 0.9670 | 28000 | 0.0033        |
| 0.9843 | 28500 | 0.0035        |
| 1.0016 | 29000 | 0.0032        |
| 1.0188 | 29500 | 0.0028        |
| 1.0361 | 30000 | 0.0028        |
| 1.0534 | 30500 | 0.0027        |
| 1.0706 | 31000 | 0.0029        |
| 1.0879 | 31500 | 0.0027        |
| 1.1052 | 32000 | 0.0027        |
| 1.1224 | 32500 | 0.0029        |
| 1.1397 | 33000 | 0.0028        |
| 1.1570 | 33500 | 0.0026        |
| 1.1742 | 34000 | 0.0027        |
| 1.1915 | 34500 | 0.0025        |
| 1.2088 | 35000 | 0.0027        |
| 1.2260 | 35500 | 0.0027        |
| 1.2433 | 36000 | 0.0027        |
| 1.2606 | 36500 | 0.0027        |
| 1.2778 | 37000 | 0.0027        |
| 1.2951 | 37500 | 0.0026        |
| 1.3124 | 38000 | 0.0026        |
| 1.3296 | 38500 | 0.0026        |
| 1.3469 | 39000 | 0.0026        |
| 1.3642 | 39500 | 0.0026        |
| 1.3815 | 40000 | 0.0027        |
| 1.3987 | 40500 | 0.0027        |
| 1.4160 | 41000 | 0.0027        |
| 1.4333 | 41500 | 0.0026        |
| 1.4505 | 42000 | 0.0026        |
| 1.4678 | 42500 | 0.0024        |
| 1.4851 | 43000 | 0.0026        |
| 1.5023 | 43500 | 0.003         |
| 1.5196 | 44000 | 0.0026        |
| 1.5369 | 44500 | 0.0028        |
| 1.5541 | 45000 | 0.0027        |
| 1.5714 | 45500 | 0.0025        |
| 1.5887 | 46000 | 0.0025        |
| 1.6059 | 46500 | 0.0024        |
| 1.6232 | 47000 | 0.0026        |
| 1.6405 | 47500 | 0.0025        |
| 1.6577 | 48000 | 0.0024        |
| 1.6750 | 48500 | 0.0027        |
| 1.6923 | 49000 | 0.0025        |
| 1.7095 | 49500 | 0.0025        |
| 1.7268 | 50000 | 0.0026        |
| 1.7441 | 50500 | 0.0025        |
| 1.7614 | 51000 | 0.0026        |
| 1.7786 | 51500 | 0.0025        |
| 1.7959 | 52000 | 0.0025        |
| 1.8132 | 52500 | 0.0026        |
| 1.8304 | 53000 | 0.0024        |
| 1.8477 | 53500 | 0.0025        |
| 1.8650 | 54000 | 0.0025        |
| 1.8822 | 54500 | 0.0024        |
| 1.8995 | 55000 | 0.0025        |
| 1.9168 | 55500 | 0.0024        |
| 1.9340 | 56000 | 0.0025        |
| 1.9513 | 56500 | 0.0025        |
| 1.9686 | 57000 | 0.0025        |
| 1.9858 | 57500 | 0.0024        |
| 2.0031 | 58000 | 0.0023        |
| 2.0204 | 58500 | 0.002         |
| 2.0376 | 59000 | 0.0019        |
| 2.0549 | 59500 | 0.002         |
| 2.0722 | 60000 | 0.002         |
| 2.0894 | 60500 | 0.0019        |
| 2.1067 | 61000 | 0.002         |
| 2.1240 | 61500 | 0.002         |
| 2.1413 | 62000 | 0.002         |
| 2.1585 | 62500 | 0.002         |
| 2.1758 | 63000 | 0.002         |
| 2.1931 | 63500 | 0.0021        |
| 2.2103 | 64000 | 0.0019        |
| 2.2276 | 64500 | 0.002         |
| 2.2449 | 65000 | 0.002         |
| 2.2621 | 65500 | 0.002         |
| 2.2794 | 66000 | 0.002         |
| 2.2967 | 66500 | 0.0021        |
| 2.3139 | 67000 | 0.002         |
| 2.3312 | 67500 | 0.0019        |
| 2.3485 | 68000 | 0.002         |
| 2.3657 | 68500 | 0.0019        |
| 2.3830 | 69000 | 0.002         |
| 2.4003 | 69500 | 0.002         |
| 2.4175 | 70000 | 0.0021        |
| 2.4348 | 70500 | 0.002         |
| 2.4521 | 71000 | 0.002         |
| 2.4693 | 71500 | 0.0021        |
| 2.4866 | 72000 | 0.002         |
| 2.5039 | 72500 | 0.0019        |
| 2.5212 | 73000 | 0.002         |
| 2.5384 | 73500 | 0.0019        |
| 2.5557 | 74000 | 0.0018        |
| 2.5730 | 74500 | 0.002         |
| 2.5902 | 75000 | 0.002         |
| 2.6075 | 75500 | 0.002         |
| 2.6248 | 76000 | 0.0021        |
| 2.6420 | 76500 | 0.0019        |
| 2.6593 | 77000 | 0.002         |
| 2.6766 | 77500 | 0.002         |
| 2.6938 | 78000 | 0.0019        |
| 2.7111 | 78500 | 0.002         |
| 2.7284 | 79000 | 0.0022        |
| 2.7456 | 79500 | 0.002         |
| 2.7629 | 80000 | 0.0019        |
| 2.7802 | 80500 | 0.002         |
| 2.7974 | 81000 | 0.002         |
| 2.8147 | 81500 | 0.0019        |
| 2.8320 | 82000 | 0.002         |
| 2.8492 | 82500 | 0.0019        |
| 2.8665 | 83000 | 0.002         |
| 2.8838 | 83500 | 0.0018        |
| 2.9011 | 84000 | 0.002         |
| 2.9183 | 84500 | 0.0019        |
| 2.9356 | 85000 | 0.002         |
| 2.9529 | 85500 | 0.002         |
| 2.9701 | 86000 | 0.0019        |
| 2.9874 | 86500 | 0.002         |

</details>

### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 4.1.0
- Transformers: 4.53.3
- PyTorch: 2.6.0+cu124
- Accelerate: 1.9.0
- Datasets: 4.0.0
- Tokenizers: 0.21.2

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