# Round 1 diagnostics

## Hyper-parameter search (our model, X + signal2)

```
 learning_rate  max_leaf_nodes  max_iter  logloss      auc
          0.05               7       200 0.291040 0.734018
          0.05               7       400 0.291040 0.734018
          0.05               7       800 0.291040 0.734018
          0.05              15       200 0.291205 0.733736
          0.05              15       400 0.291205 0.733736
          0.05              15       800 0.291205 0.733736
          0.05              31       200 0.291697 0.732337
          0.05              31       400 0.291697 0.732337
          0.05              31       800 0.291697 0.732337
```


## Model quality (out of sample)

```
X only                       AUC=0.7174  logloss=0.2951  brier=0.0848
X + signal2 (us)             AUC=0.7349  logloss=0.2908  brier=0.0838
X + signal1 (lender 1)       AUC=0.7353  logloss=0.2908  brier=0.0837
X + signal3 (lender 3)       AUC=0.7363  logloss=0.2904  brier=0.0836
X + all signals (full info)  AUC=0.7603  logloss=0.2828  brier=0.0817
```


## Calibration of our model (deciles)

```
         n  predicted  realized
bin                            
0    10003     0.0066    0.0062
1    10008     0.0215    0.0212
2     9993     0.0354    0.0332
3    10012     0.0543    0.0532
4     9987     0.0770    0.0767
5    10020     0.1013    0.1019
6    10055     0.1230    0.1220
7     9948     0.1445    0.1429
8    10077     0.1794    0.1821
9     9897     0.2527    0.2563
```


## Hidden information U = p_full - p_ours

```
std(U) = 0.0446
by income segment:
0         0.0714
1-5k      0.0504
10-20k    0.0246
20-40k    0.0084
40k+      0.0027
5-10k     0.0517
```


## Chosen pricing rule (segment margins)

```
{
  "margins": {
    "0": 0.04,
    "1-5k": 0.035,
    "10-20k": 0.03,
    "20-40k": 0.03,
    "40k+": 0.03,
    "5-10k": 0.035
  },
  "tau": {
    "0": 0.12,
    "1-5k": 0.12,
    "10-20k": 0.08,
    "20-40k": 0.25,
    "40k+": 0.25,
    "5-10k": 0.1
  },
  "tail_rate": 0.828,
  "mean_expected_profit_M": 6.574,
  "worst_scenario_profit_M": 1.238
}
```


## Strategy: naive: break-even + 3%, lend to everyone

```
                           share      loans  profit_exp_M  profit_real_M  our_p2_on_won  realized_default  mean_rate  lender1_profit_M  lender3_profit_M
base / base                0.327  32666.333         1.671          1.521          0.081             0.100      0.123             2.417             2.547
aggressive / aggressive    0.162  16165.333        -1.641         -2.193          0.089             0.123      0.132            -3.080            -3.255
cautious / cautious        0.575  57504.333         7.711          8.052          0.082             0.093      0.126             4.019             3.626
no_reject / no_reject      0.325  32509.667         1.975          1.843          0.080             0.098      0.121             2.217             2.366
slide / slide              0.251  25056.000         1.142          0.627          0.067             0.088      0.105            -0.793            -0.494
slide_norej / slide_norej  0.245  24546.667         2.118          1.703          0.063             0.081      0.099            -1.536            -1.065
smart / smart              0.413  41257.667         3.792          4.232          0.090             0.104      0.134             3.703             3.562
base_net / base_net        0.327  32666.333         1.671          1.521          0.081             0.100      0.123             2.417             2.547
base / aggressive          0.217  21735.667        -0.978         -1.284          0.088             0.116      0.131            -1.005            -1.263
cautious / slide           0.359  35914.333         2.339          1.593          0.072             0.092      0.113             0.548             2.136
smart / base               0.364  36370.333         2.419          2.403          0.085             0.102      0.128             2.446             3.649
loose / tight              0.313  31275.333         0.946          0.921          0.085             0.105      0.128             2.218             1.242
```


## Strategy: naive: break-even + 3%, no offer if PD > 25%

```
                           share      loans  profit_exp_M  profit_real_M  our_p2_on_won  realized_default  mean_rate  lender1_profit_M  lender3_profit_M
base / base                0.324  32370.000         2.341          2.173          0.079             0.097      0.120             2.409             2.544
aggressive / aggressive    0.161  16067.667        -1.362         -1.928          0.088             0.121      0.130            -3.218            -3.340
cautious / cautious        0.558  55771.333         8.581          9.010          0.075             0.085      0.116             4.019             3.626
no_reject / no_reject      0.324  32359.333         2.346          2.207          0.079             0.096      0.120             2.027             2.245
slide / slide              0.248  24772.000         1.801          1.270          0.064             0.083      0.102            -0.793            -0.494
slide_norej / slide_norej  0.245  24532.667         2.166          1.742          0.063             0.080      0.099            -1.555            -1.084
smart / smart              0.408  40817.667         4.535          4.850          0.088             0.101      0.131             3.704             3.793
base_net / base_net        0.324  32370.000         2.341          2.173          0.079             0.097      0.120             2.456             2.703
base / aggressive          0.216  21552.667        -0.532         -0.894          0.086             0.113      0.129            -1.014            -1.511
cautious / slide           0.353  35287.333         3.248          2.735          0.069             0.086      0.107             0.548             2.136
smart / base               0.360  36029.333         3.128          3.084          0.083             0.099      0.125             2.615             3.649
loose / tight              0.308  30837.667         1.677          1.597          0.082             0.101      0.124             2.189             1.242
```


## Strategy: naive + safety net 0.989 instead of no offer

```
                           share      loans  profit_exp_M  profit_real_M  our_p2_on_won  realized_default  mean_rate  lender1_profit_M  lender3_profit_M
base / base                0.327  32654.000         2.514          2.376          0.081             0.100      0.128             2.409             2.544
aggressive / aggressive    0.161  16067.667        -1.362         -1.928          0.088             0.121      0.130            -3.218            -3.340
cautious / cautious        0.575  57504.333        14.087         14.415          0.082             0.093      0.142             4.019             3.626
no_reject / no_reject      0.324  32359.333         2.346          2.207          0.079             0.096      0.120             2.027             2.245
slide / slide              0.251  25056.000         1.975          1.473          0.067             0.088      0.112            -0.793            -0.494
slide_norej / slide_norej  0.245  24532.667         2.166          1.742          0.063             0.080      0.099            -1.555            -1.084
smart / smart              0.409  40919.667         4.607          4.871          0.089             0.102      0.133             3.713             3.762
base_net / base_net        0.325  32472.000         2.413          2.194          0.080             0.098      0.123             2.465             2.672
base / aggressive          0.216  21552.667        -0.532         -0.894          0.086             0.113      0.129            -1.014            -1.511
cautious / slide           0.359  35914.333         4.409          3.586          0.072             0.092      0.123             0.548             2.136
smart / base               0.362  36224.000         3.256          3.232          0.085             0.101      0.130             2.465             3.649
loose / tight              0.311  31132.667         1.975          1.869          0.084             0.104      0.132             2.189             1.242
```


## Strategy: OPTIMIZED rule

```
                           share      loans  profit_exp_M  profit_real_M  our_p2_on_won  realized_default  mean_rate  lender1_profit_M  lender3_profit_M
base / base                0.217  21653.000         5.336          5.059          0.054             0.068      0.110             3.174             3.173
aggressive / aggressive    0.091   9139.000         1.238          1.210          0.049             0.064      0.085            -3.992            -4.037
cautious / cautious        0.427  42670.667        24.929         24.757          0.066             0.076      0.169             6.485             6.243
no_reject / no_reject      0.208  20817.000         4.610          4.663          0.046             0.053      0.081             2.345             2.455
slide / slide              0.194  19418.333         4.716          4.357          0.051             0.067      0.109            -1.066            -0.888
slide_norej / slide_norej  0.186  18582.333         3.990          3.961          0.043             0.050      0.077            -2.172            -1.853
smart / smart              0.247  24721.000         6.470          6.427          0.057             0.067      0.111             6.225             6.293
base_net / base_net        0.217  21653.000         5.336          5.059          0.054             0.068      0.110             3.174             3.173
base / aggressive          0.126  12594.333         2.075          2.169          0.050             0.061      0.085            -1.115            -2.112
cautious / slide           0.272  27170.333         9.301          8.710          0.056             0.071      0.130             0.334             2.741
smart / base               0.230  22964.000         5.767          5.612          0.055             0.068      0.111             3.336             5.499
loose / tight              0.200  20039.667         5.122          4.989          0.054             0.067      0.111             2.630             1.905
```


## Summary across scenarios (realized profit on past loans)

```
                                              mean profit (M EUR)  worst scenario (M EUR)  mean share
naive: break-even + 3%, lend to everyone                    1.745                  -2.193       0.323
naive: break-even + 3%, no offer if PD > 25%                2.335                  -1.928       0.319
naive + safety net 0.989 instead of no offer                2.929                  -1.928       0.322
OPTIMIZED rule                                              6.415                   1.210       0.218
```


## Optimized rule by income segment (scenario base / base)

```
         applicants  offered  share_won  mean_rate_offered  our_p2_on_won  realized_default_on_won  profit_M
segment                                                                                                     
0             17035      1.0      0.079              0.796          0.180                    0.298     0.271
1-5k           6125      1.0      0.125              0.678          0.106                    0.112     0.268
10-20k        25923      1.0      0.319              0.217          0.046                    0.053     1.733
20-40k        12780      1.0      0.339              0.054          0.022                    0.023     1.241
40k+           8233      1.0      0.332              0.036          0.006                    0.007     0.800
5-10k         29904      1.0      0.144              0.608          0.080                    0.100     0.679
```


## Submission: output/round1/rates_round1.csv

```
applicants: 100000   offers: 100000   no offer: 0   at safety-net rate 0.828: 45582
mean PD (our model): 0.0997

by segment:
         applicants  mean_p2  offered  median_rate  min_rate  max_rate
segment                                                               
0             17159   0.2009      1.0       0.8280    0.1666    0.8280
1-5k           6142   0.1501      1.0       0.8280    0.1070    0.8280
10-20k        25766   0.0583      1.0       0.0858    0.0382    0.8280
20-40k        12829   0.0234      1.0       0.0525    0.0351    0.1010
40k+           8234   0.0055      1.0       0.0342    0.0304    0.0891
5-10k         29870   0.1255      1.0       0.8280    0.0555    0.8280
```
