# cogsci-2023-multiplechoice
Companion repository for the 2023 article "Effect of Response Format on Syllogistic Reasoning" published in the proceedings of the 45rd Annual Meeting of the Cognitive Science Society.

## Overview

- `analysis`: Contains the scripts for performing the analyses and generating the plots.
- `analyses/create_mfa_dict.py`: Creates a JSON dictionary for the MFA model (Creates `model_eval/individual/mfa/mfa.json`).
- `analyses/create_relevant.py`: Creates JSON dictionaries for the relevant responses ("liable pooled conclusions"). They are stored in the data folder.
- `analyses/figure_effect_implicature.py`: Calculates the occurences of implicatures and the statistics for the figure effect.
- `analyses/plot_comparison.py`: Plots the comparison of the patterns of all 3 datasets (Figure 2).
- `analyses/plot_multi.py`: Plots the response patterns with the most frequent selection of the multiple choice data (Figure 1).
- `analyses/pooled_analysis.py`: Compares the 3 datasets (free choice, single choice and multiple choice) on the basis of liable pooled conclusions (Jaccard Similariy, number of intersections and average number of responses).
- `analyses/rmse_mfa.py`: Calculates the RMSE and the MFA congruency.
- `analyses/some_all.py`: Prints the ratio of participants noting that "Some" entails "All".

- `data`: Contains the data for the analyses.
- `data/Khemlani2012.csv`: Percentages of responses for all syllogism in the dataset with free responses (as reported by Khemlani & Johnson-Laird (2012) in the paper "Theories of the syllogism: A meta-analysis"). Data was obtained from the tables provided on [OSF](https://osf.io/rwyk5/ ).
- `data/Khemlani_liable.csv`: Liable pooled conclusions as taken from the free response data (Khemlani & Johnson-Laird, 2012). Retrieved from [OSF](https://osf.io/rwyk5/ ).
- `data/multiple_choice.csv`: Dataset containing the multiple-choice responses to all 64 syllogisms.
- `data/Ragni2016.csv`: Single-choice dataset containing the responses of 139 participants provided with the [CCOBRA framework](https://github.com/CognitiveComputationLab/ccobra ).
- `data/relevant_khem.json`: JSON containing the liable pooled conclusions for the free response dataset.
- `data/relevant_multi.json`: JSON containing the liable pooled conclusions for the multiple-choice dataset.
- `data/relevant_ragni.json`: JSON containing the liable pooled conclusions for the single-choice dataset.

- `model_eval`: Contains the models and scripts for the model evaluation.
- `model_eval/individual`: Contains the models and scripts for the model evaluation predicting individual participants.
- `model_eval/individual/mfa`: Contains the MFA model and the respective response dictionary.
- `model_eval/individual/mfa/mfa.json`: The most-frequent answers represented as JSON dictionary.
- `model_eval/individual/mfa/mfaModel.py`: CCOBRA model implementing the MFA.
- `model_eval/individual/mfa`: Contains the MFA model and the respective response dictionary.
- `model_eval/individual/models`: Contains a CCOBRA model predicting based on a given prediction-table and the respective model-prediction tables according to the analysis performed by Khemlani & Johnson-Laird in the 2012 paper "Theories of the syllogism: A meta-analysis".
- `model_eval/individual/models/csvModel.py`: General CCOBRA model for predicting based on a CSV prediction table.
- `model_eval/individual/mreasoner`: CCOBRA model of mReasoner based on a cache file. Based on the [implementation](https://github.com/nriesterer/cogsci-individualization ) provided with the 2020 paper "Do Models Capture Individuals? Evaluating Parameterized Models for Syllogistic Reasoning" by Riesterer, Brand & Ragni.
- `model_eval/individual/mreasoner/mreasoner_jacc.py`: Adaptation of the CCOBRA mReasoner implementation by Riesterer, Brand & Ragni (2020) that supports multiple-choice and fits with respect to the jaccard coefficient.
- `model_eval/individual/mreasoner/cache`: Contains the cache file for mReasoner.
- `model_eval/individual/mreasoner/cache/nec_cache.npy`: Cache containing the predictions of mReasoner for all 64 syllogisms and response options by querying with "Is it necessary?".
- `model_eval/individual/phm`: Contains the CCOBRA PHM-implementation by Riesterer, Brand & Ragni (2020) adapted to multiple-choice and the jaccard coefficient.
- `model_eval/individual/phm/phm.py`: PHM base implementation by Riesterer, Brand & Ragni (2020).
- `model_eval/individual/phm/phm_ccobra_jacc.py`: CCOBRA model for PHM, adapted to multiple-choice and the jaccard coefficient.
- `model_eval/individual/results`: Contains the results for of the CCOBRA evaluation.
- `model_eval/individual/results/jaccard.csv`: CSV dataset containing the results for of the CCOBRA evaluation.
- `model_eval/jaccard.py`: CCOBRA comparator implementing the Jaccard coefficient for the analysis.
- `model_eval/multi_jaccard.json`: CCOBRA benchmark file specifying the evaluation configuration.
- `model_eval/plot_performance_jaccard.py`: Plots the results of the CCOBRA evaluation (dataset is stored in `model_eval/results/jaccard.csv`).
- `model_eval/models`: Contains model prediction tables according to the analysis performed by Khemlani & Johnson-Laird in the 2012 paper "Theories of the syllogism: A meta-analysis".
- `model_eval/models/Atmosphere.csv`: The prediction table for the Atmosphere theory.
- `model_eval/models/Conversion.csv`: The prediction table for the Conversion hypothesis.
- `model_eval/models/Matching.csv`: The prediction table for the Matching theory.
- `model_eval/models/MMT.csv`: The prediction table for the Mental Model Theory.
- `model_eval/models/PHM.csv`: The prediction table for the Probability Heuristics Model.
- `model_eval/models/PSYCOP.csv`: The prediction table for the Psychology of Proof model.
- `model_eval/models/VerbalModels.csv`: The prediction table for the Verbal Models.
- `model_eval/model_eval_agg.py`: Script performing the aggregated analysis based on the model prediction tables calculating accuracy, hits, rejections, jaccard coefficient, as well as the correlation between rejection and the number of responses.
- `model_eval/sample.py`: Script calculating the values for the theoretical comparison of accuracy and jaccard coefficient.

## Dependencies

- Python 3
    - CCOBRA
    - pandas
    - numpy
    - seaborn

## Run analysis scripts

All analysis scripts can be run without any arguments. To run a specific script, use:

```
cd /path/to/repository/analysis
$> python [script].py
```

The plots are then saved in the same directory.

## Run model analyses

The aggregated analysis and the plotting of the results can be run like the analysis scripts.
For the CCOBRA evaluation, the benchmark file can be executed by the following commands:

```
cd /path/to/repository/model_eval/individual
$> ccobra multi_jaccard.json
```

A HTML file containing the results will be placed in the same directory. The results can be saved as a CSV file from there (resembling the file stored in the results subfolder).

## References
Brand, D., & Ragni, M. (In Press). Effect of Response Format on Syllogistic Reasoning. In Proceedings of the 45rd Annual Meeting of the Cognitive Science Society.

Khemlani, S. S., & Johnson-Laird, P. N. (2012). Theories of the syllogism: A meta-analysis. Psychological Bulletin, 138(3), 427–457. [OSF](https://osf.io/rwyk5/

Riesterer, N., Brand, D., & Ragni, M. (2020). Do Models Capture Individuals? Evaluating Parameterized Models for Syllogistic Reasoning. In Proceedings of the 42nd Annual Conference of the Cognitive Science Society. [GitHub](https://github.com/nriesterer/cogsci-individualization

