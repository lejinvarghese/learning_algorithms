## General

```bash
python3 processor.py --sample_size 1000
python3 processor.py --sample_size 1000 --subset_name triplets
```


### Results
 - bert-base-uncased: 0.9775
 - thenlper/gte-small: val: 0.9922, test: 0.9664

 ## Datasets
- [x] Home Depot
- [x] Amazon
- [x] Google
- [x] Wayfair
- [ ] Crowdflower
- [ ] Walmart

## Improvements
- [ ] Add more Amazon metadata
- [ ] Add more Google metadata
- [ ] Normalize scores between datasets

Home Depot: Queries: 11199, Documents: 39726.
Amazon: Queries: 38009, Documents: 34066.
Wayfair: Queries: 474, Documents: 26224.
Google: Queries: 1025, Documents: 46524.
Crowdflower: Queries: 261, Documents: 9912.
