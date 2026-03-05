## General

```bash
python3 processor.py --sample_size 1000
python3 processor.py --sample_size 1000 --subset_name pairs
python3 processor.py --sample_size 100000 --subset_name "triplets" --chunk_size 10000 --chunk_suffix "triplets"
```


### Results
 - bert-base-uncased: 0.9775
 - thenlper/gte-small: val: 0.9922, test: 0.9664

 ## Datasets
- [x] Home Depot
- [x] Amazon
- [x] Google
- [x] Wayfair
- [x] Crowdflower
- [ ] Walmart

## Improvements
- [ ] Add more Amazon metadata
- [ ] Add more Google metadata
- [ ] Normalize scores between datasets


### Sources
| Dataset | Repo ID | Source |
|-------------|---------|--------|
| Google | Marqo/marqo-GS-10M | Google Shopping |
| Amazon | tasksource/esci | Amazon ESCI |
| Wayfair | napsternxg/wands | Wayfair |
| Home Depot | bstds/home_depot | Home Depot |
| Crowdflower | napsternxg/kaggle_crowdflower_ecommerce_search_relevance | Crowdflower |

### Train

| Dataset     | Queries | Documents | Pairs    |
|-------------|---------|-----------|----------|
| Google      | 77,288  | 2,202,907 | 3,926,764|
| Amazon      | 99,408  | 985,476   | 1,420,372|
| Wayfair     | 477     | 38,854    | 140,068  |
| Home Depot  | 11,795  | 54,360    | 74,067   |
| Crowdflower | 261     | 9,912     | 10,158   |

### Test

| Dataset     | Queries | Documents | Pairs    |
|-------------|---------|-----------|----------|
| Google      | 19,564  | 748,386   | 981,204  |
| Amazon      | 30,947  | 364,004   | 434,234  |
| Wayfair     | 477     | 25,317    | 46,690   |


## Remote

```bash
nano ~/.ssh/config
```


```bash
sky launch -c datasets datasets.yml \
  --env HF_TOKEN=$HF_TOKEN \
  --idle-minutes-to-autostop 10 \
  --down
```
