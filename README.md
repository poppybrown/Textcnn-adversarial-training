# Textcnn-adversarial-training

The repo is the tensorflow implementation for the adversarial training on hotel reviews sentiment dataset with FGSM、PGD、FREEAT.

And the textcnn code refer the repo https://github.com/norybaby/sentiment_analysis_textcnn

# Train Model

1. if you want to train with **baseline**,you can use this `train(None)` by default parameters.if want to change parameters,you can change parametes as what you want.
2. if you want to train with **fgsm**,you can use this `train(None, mode='fgsm')` by default parameters.if want to change parameters,you can change parametes as what you want.
3. if you want to train with **pgd**,you can use this `train(None, mode='pgd')` by default parameters.if want to change parameters,you can change parametes as what you want.
4. if you want to train with **free**,you can use this `train(None, is_free=True, mode='free')` by default parameters.if want to change parameters,you can change parametes as what you want.

# Result

training with 1 epoch,and the performance on test set result is below

| method                                    | micro-recall | micro-precison | micro-f1 |
| ----------------------------------------- | ------------ | -------------- | -------- |
| text-baseline                             | 89.39%       | 89.39%         | 89.39%   |
| fgsm(alpha=10 / 255)                      | 75%          | 75%            | 75%      |
| pgd(alpha=10 / 255,epsilon=8 / 255，K=5)  | 75%          | 75%            | 75%      |
| free(alpha=10 / 255,epsilon=8 / 255，K=5) | 75%          | 75%            | 75%      |

From the result,we can see that adversarial training with fgsm、pgd、free reduce performance，i guess that the value of alpha is not proper.

# Note

If there is error in the program, please point it out

# Reference

[1] EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

[2] ADVERSARIAL TRAINING METHODS FOR SEMI-SUPERVISED TEXT CLASSIFICATION

[3] Towards Deep Learning Models Resistant to Adversarial Attacks

[4] Adversarial Training for Free!

[5] You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle

[6] FREELB: ENHANCED ADVERSARIAL TRAINING FOR NATURAL LANGUAGE UNDERSTANDING

[7] FAST IS BETTER THAN FREE: REVISITING ADVERSARIAL TRAINING