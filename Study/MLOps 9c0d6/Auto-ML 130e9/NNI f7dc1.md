# NNI

- *Experiment*: One task of, for example, finding out the best hyperparameters of a model, finding out the best neural network architecture, etc. It consists of trials and AutoML algorithms.
- *Search Space*: The feasible region for tuning the model. For example, **the value range of each hyperparameter.**
- *Configuration*: An instance from the search space, that is, each hyperparameter has a specific value.
- *Trial*: An individual attempt at applying a new configuration (e.g., a set of hyperparameter values, a specific neural architecture, etc.). Trial code should be able to run with the provided configuration.
- *Tuner*: An AutoML algorithm, which **generates a new configuration for the next try.** A new trial will run with this configuration.
- *Assessor*: Analyze a trial’s intermediate results (e.g., periodically evaluated accuracy on test dataset) **to tell whether this trial can be early stopped or not.**
- *Training Platform*: Where trials are executed. Depending on your experiment’s configuration, it could be your local machine, or remote servers, or large-scale training platform (e.g., OpenPAI, Kubernetes).

**Experiment 실행**

→ Tuner가 search space를 받아 configuration을 생성.

→ 생성된 configuration을 training platform에 전달 (local machine, remote machine, training cluster)

→ 성능이 Tuner로 다시 전달

→ 새로운 configuration을 생성 후 다시 전달.

즉, 매 Experiment 시, `Search Space` 정의 및 몇 줄의 코드 수정만 해주면 됨.

**Search Space**

[Search Space - An open source AutoML toolkit for neural architecture search, model compression and hyper-parameter tuning (NNI v2.1)](https://nni.readthedocs.io/en/stable/Tutorial/SearchSpaceSpec.html)

1. Search Space에서 dictionary 형태로 넣고 싶은 경우는 "_name" 이라는 key 값만 추가 해주면됨. ex)`{"_type": "choice", "_value": options}` 
→ `{"_type": "choice", "_name" : 'name', "_value": options}`

2.  `{"_type": "quniform", "_value": [low, high, q]}`
ex 1) `_value` specified as `[0, 10, 2.5]`
→ `[0, 2.5, 5.0, 7.5, 10.0]`
ex 2) `_value` specified as `[2, 10, 5]`
→ `[2, 5, 10]`

**Update model codes** (**Hyperparameter** 방법)

1) Search Space file 이용

```python
useAnnotation: false
searchSpacePath: /path/to/your/search_space.json

->
### searchSpace file 내용
{
    "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
    "conv_size":{"_type":"choice","_value":[2,3,5,7]},
    "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
    "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
}
```

2) Annotation (주석 다는 방식)

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
+   """@nni.variable(nni.choice(50, 250, 500), name=batch_size)"""
    batch_size = 128
    for i in range(10000):
        batch = mnist.train.next_batch(batch_size)
+       """@nni.variable(nni.choice(0.1, 0.5), name=dropout_rate)"""
        dropout_rate = 0.5
        mnist_network.train_step.run(feed_dict={mnist_network.images: batch[0],
                                                mnist_network.labels: batch[1],
                                                mnist_network.keep_prob: dropout_rate})
        if i % 100 == 0:
            test_acc = mnist_network.accuracy.eval(
                feed_dict={mnist_network.images: mnist.test.images,
                            mnist_network.labels: mnist.test.labels,
                            mnist_network.keep_prob: 1.0})
+           """@nni.report_intermediate_result(test_acc)"""

    test_acc = mnist_network.accuracy.eval(
        feed_dict={mnist_network.images: mnist.test.images,
                    mnist_network.labels: mnist.test.labels,
                    mnist_network.keep_prob: 1.0})
+   """@nni.report_final_result(test_acc)"""
```

Annotation → True

```python
useAnnotation: true
```

**Define Experiment (yaml file)**

[Experiment Config Reference - An open source AutoML toolkit for neural architecture search, model compression and hyper-parameter tuning (NNI v2.1)](https://nni.readthedocs.io/en/stable/Tutorial/ExperimentConfig.html)

```python
authorName: # The name of the author who create the experiment.

experimentName: # The name of the experiment created.

trialConcurrency: # Specifies the max num of trial jobs run simultaneously. _ gpu number 초과시, put into a queue

maxExecDuration: # specifies the max duration time of an experiment. 

								 # The unit of the time is {sm, h, d}, which means {seconds, minutes, hours, days}.
maxTrialNum:  # Specifies the max number of trial jobs created by NNI, including succeeded and failed jobs.
#choice: local, remote, pai, kubeflow

trainingServicePlatform: # Specifies the platform to run the experiment, including local, remote, pai, kubeflow, frameworkcontroller.
# local : run an experiment on local ubuntu machine.
# remote : submit trial jobs to remote ubuntu machines, and machineList field should be filed in order to set up SSH connection to remote machine.
# pai : submit trial jobs to OpenPAI of Microsoft. For more details of pai configuration, please refer to Guide to PAI Mode
# kubeflow : submit trial jobs to kubeflow, NNI support kubeflow based on normal kubernetes and azure kubernetes. For detail please refer to Kubeflow Docs
# adl : submit trial jobs to AdaptDL, NNI support AdaptDL on Kubernetes cluster. For detail please refer to AdaptDL Docs

searchSpacePath: # SearchSpace file path
#choice: true, false, default: false
useAnnotation: # Use annotation
#choice: true, false, default: false
multiThread: # Enable multi-thread mode for dispatcher. If multiThread is enabled, dispatcher will start a thread to process each command from NNI Manager.
tuner:
# 둘중 하나만 선택
# 1) NNI sdk tuner : builtinTunername, classArgs
# 2) own tuner file : codeDirectory, classFileName, className, classArgs
  #choice: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    #choice: maximize, minimize
    optimize_mode:
  gpuIndices:
trial:
  command: # Specifies the command to run trial process.
  codeDir: # Specifies the directory of the own trial file. Files in the directory will be uploaded in PAI mode.
  gpuNum: # Specifies the num of gpu to run the trial process. Default value is 0.

#machineList can be empty if the platform is local
machineList:
  - ip:
    port:
    username:
    passwd:

```

### Cifar10 - Normal

![NNI%20f7dc1/Untitled.png](NNI%20f7dc1/Untitled.png)