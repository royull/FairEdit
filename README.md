# FairEdit

In this work we introduce FairEdit, the unexplored method of edge addition, accompanied by deletion, to promote fairness. FairEdit performs efficient edge editing by leveraging gradient information of a fairness loss to find edges that improve fairness. We find that FairEdit outperforms standard training for many data sets and GNN methods, while performing comparably to many state-of-the-art methods, demonstrating FairEdit's ability to improve fairness across many domains and models.

## Relevent Publication

FairEdit: Preserving Fairness in Graph Neural Networks through Greedy Graph Editing
- [Arxiv](https://arxiv.org/abs/2201.03681)
- Cite
``` bibtex
@misc{loveland2022fairedit,
      title={FairEdit: Preserving Fairness in Graph Neural Networks through Greedy Graph Editing}, 
      author={Donald Loveland and Jiayi Pan and Aaresh Farrokh Bhathena and Yiyang Lu},
      year={2022},
      eprint={2201.03681},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Reproduce the results
### Requirements

Use ```environment.yml``` to build the environment
### FairEdit & BruteFoce
Run ```adjusted_training.py``` with specified arguments to do traning with various setups. Model weights are saved to weights folder and evaluation metricss are saved to results. 

Arguments:
- ```--model``` can be ```['gcn', 'sage', 'appnp']``` 
- ```--dataset``` can be ```['german', 'credit', 'bail']```
- ```--training_method``` can be ```['fairedit','brute']```
- Hyper-parameters also need to be specified as shown in the example below

Sample code:
```
python adjusted_training.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --training_method fairedit --dataset german --seed 1
```
### Standard Training & Nifty
The same ```adjusted_traning.py``` can also be used to run standard training and [NIFTY](https://arxiv.org/abs/2102.13186)
- Specifically, you just need to change traning method to ```['standard', 'nifty']```

### FairWalk
Code to run [FairWalk](https://www.ijcai.org/proceedings/2019/0456.pdf) can be found in ```models_to_compare_against/FairWalk```