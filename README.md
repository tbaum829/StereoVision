# ECE 570 Term Paper Implementation: A Review of Stereo Disparity Estimation Algorithms Using Differentiable PatchMatch

### Table of Contents  
[Classical Stereo](#ClassicalStereo)  
[My Algorithm](#MyAlgorithm)  
[Differentiable PatchMatch](#DifferentiablePatchMatch)  
[Experiments/Testing](#ExperimentsTesting)  
[Acknowledgements](#Acknowledgements)  

<a name="ClassicalStereo"></a>
## Classical Stereo
The classical approach to solving the problem of depth estimation for stereo imagery is to create a mapping between patches, *A*, centered around each pixel in the left image with patches, *B*, in its paired right image while minimizing some cost function, *f* (*A*, *B*), typically a simple error sum of squares (SSE). The horizontal pixel disparity between paired patches *A* and *B* can then be used to calculate the absolute distance between the camera and the object by using information about the positioning and resolution of the two cameras. 

The entirety of the Classical Stereo algorithm is contained within the *classicStereo.py* file and can either be tested by running the file as main or by being imported as a module and running the *main()* function with the following parameters:
1. **left_path:** The path to the left source image.
2. **right_path:** The path to the right source image.
3. **outfile:** The output path for the resulting disparity map.
4. **disparity_range:** The max disparity expected for the given stereo image pair.

<a name="MyAlgorithm"></a>
## My Algorithm
My algorithm applies [PatchMatch](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf) in the context of disparity estimation using stereo imaging similar to what was proposed by [DeepPruner](http://www.cs.toronto.edu/~slwang/deeppruner.pdf). Instead of using PatchMatch twice to narrow the search field of possible disparities, my algorithm only runs PatchMatch once while still including the random search phase to preserve the efficacy of escaping local minima.

The entirety of my algorithm is contained within the *patchMatch.py* file and can either be tested by running the file as main or by being imported as a module and running the *main()* function with the following parameters:
1. **left_path:** The path to the left source image.
2. **right_path:** The path to the right source image.
3. **outfile:** The output path for the resulting disparity map.
4. **disparity_range:** The max disparity expected for the given stereo image pair.
4. **iterations:** The number of iterations you would like the algorithm to complete.

<a name="DifferentiablePatchMatch"></a>
## Differentiable PatchMatch
DeepPruner presents a differentiable version of [PatchMatch](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf) that removes the random search phase and replaces the *arg max* function with a [soft version](https://arxiv.org/pdf/1703.04309.pdf) to maintain differentiability. Also, in the propagation phase instead of alternating propagation from top left to the bottom right and from the bottom right to top left, the differentiable PatchMatch algorithm presented by [Duggal et al.](http://www.cs.toronto.edu/~slwang/deeppruner.pdf) utilizes one-hot filter banks for all four surrounding pixels.

The entirety of the Differentiable PatchMatch algorithm is contained within the *diffPatchMatch.py* file and can either be tested by running the file as main or by being imported as a module and running the *main()* function with the following parameters:
1. **left_path:** The path to the left source image.
2. **right_path:** The path to the right source image.
3. **outfile:** The output path for the resulting disparity map.
4. **disparity_range:** The max disparity expected for the given stereo image pair.
4. **iterations:** The number of iterations you would like the algorithm to complete.

<a name="ExperimentsTesting"></a>
## Experiments/Testing
I tested all three models against 6 sample stereo pairs from the [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) sample dataset. Runtimes were compared using the built-in Python Time module and the disparity maps were compared visually.

The testing script works by iterating through each each left and right stereo image pair contained within the *source* directory and produces 7 output files cooresponding to the following algorithm calls:
- Classical Stereo
- My Algorithm (1 Iteration)
- My Algorithm (2 Iterations)
- My Algorithm (5 Iterations)
- Differentiable PatchMatch (1 Iteration)
- Differentiable PatchMatch (2 Iterations)
- Differentiable PatchMatch (5 Iterations)

The output files are stored with the same names as the input files in the *output* directory within their model's cooresponding sub-directory.

The entirety of the Experiments/Testing script is contained within the *experiments.py* file and should be run as main with no arguments.

<a name="Acknowledgements"></a>
## Acknowledgements

#### Classic Stereo
I independently implemented the code for the Classical Stereo disparity estimation model.
The algorithm was inspired, however, by a 2010 article written by Chris McCormick called ["Stereo Vision Tutorial - Part I"](http://mccormickml.com/2014/01/10/stereo-vision-tutorial-part-i/).

#### My Algorithm
I independently implemented the code for my disparity estimation algorithm.
The algorithm was inspired, however, by the PatchMatch ([Barnes et al.](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf)) and DeepPruner ([Duggal et al.](http://www.cs.toronto.edu/~slwang/deeppruner.pdf)) papers.

#### Differentiable PatchMatch
The code for the Differentiable PatchMatch Model was formed by modifying existing an existing program written by Shivam Duggal that can be found [here](https://github.com/uber-research/DeepPruner/blob/master/DifferentiablePatchMatch/README.md).
Duggal et al.'s paper on DeepPruner can be found [here](http://www.cs.toronto.edu/~slwang/deeppruner.pdf).

The source code I modified to create this model was initially intended for image reconstruction of one image using approximate near neighbor patch matches from a second image. I modified the algorithm by reducing the search space to just the x-axis within the given disparity range and had the output of the algorithm only contain the resulting offsets. I also condenced the source code from several files/modules down to one for simplicity.

#### Experiments/Testing
I independently implemented the script used to test all three of the algorithms.
The source stereo images used by the testing script came from the [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) sample dataset.
