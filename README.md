# Link Prediction on PharmKG with CompGCN in Tensorflow
An implementation of **CompGCN** in [Tensorflow](https://www.tensorflow.org/). The model perform a **link prediction** task on **PharmKG** dataset. 
- Paper: https://arxiv.org/abs/1911.03082
- Author's code: https://github.com/malllabiisc/CompGCN

## Dependencies
1. Install Python3 and Tensorflow-cuda
2. Install the requirements from `requirements.txt`

## Dataset
- PharmKG: https://github.com/MindRank-Biotech/PharmKG
- Paper: https://academic.oup.com/bib/article/22/4/bbaa344/6042240
## Train model
To train the model:

`python run.py -score_func transe -opn sub -init_dim 200`

or

`python run.py -score_func distmult -opn sub -gcn_dim 150`

or

`python run.py -score_func conve -opn sub ker_sz 5`

- `-score_func` is the scoring function for link prediction: `transe`, `distmult` or `conve`
- `-opn` is the composition operation: `sub` for subtraction, `mult` for multiplication or `corr` for circular correlation
- Rest of the arguments can be listed using `python run.py -h`

## Result
The following results are obtained doing, for every score function, 5 tests with different seeds and then taking their average and standard deviation.
As we can see the **best models** are ConvE with every composition operators. 

<table>
   <tbody><tr>
      <th rowspan="2">Composition Operation
      </th><th rowspan="2">Score Function
      </th><td></td>
      <th colspan="4">Hits@N
   </th></tr>
   <tr>
      <th>MRR
      </th><th>N=1
      </th><th>N=3
      </th><th>N=10
   </th></tr>
   <tr>
      <th rowspan="3">Sub
      </th><th>TransE
      </th><td>0.143 ± 0.017</td>
      <td>0.066 ± 0.016</td>
      <td>0.159 ± 0.021</td>
      <td>0.292 ± 0.026</td>
   </tr>
   <tr>
      <th>DistMult
      </th><td>0.117 ± 0.009</td>
      <td>0.051 ± 0.006</td>
      <td>0.120 ± 0.011</td>
      <td>0.245 ± 0.023</td>
   </tr>
   <tr>
      <th>ConvE
      </th><td><b>0.170 ± 0.019</b></td>
      <td><b>0.096 ± 0.014</b></td>
      <td><b>0.186 ± 0.029</b></td>
      <td><b>0.319 ± 0.023</b></td>
   </tr>
   <tr>
      <th rowspan="3">Mult
      </th><th>TransE
      </th><td>0.143 ± 0.027</td>
      <td>0.078 ± 0.020</td>
      <td>0.141 ± 0.030</td>
      <td>0.289 ± 0.043</td>
   </tr>
   <tr>
      <th>DistMult
      </th><td>0.117 ± 0.011</td>
      <td>0.053 ± 0.010</td>
      <td>0.113 ± 0.014</td>
      <td>0.241 ± 0.030</td>
   </tr>
   <tr>
      <th>ConvE
      </th><td><b>0.158 ± 0.005</b></td>
      <td><b>0.088 ± 0.007</b></td>
      <td><b>0.169 ± 0.008</b></td>
      <td><b>0.298 ± 0.018</b></td>
   </tr>
   <tr>
      <th rowspan="3">Circular Correlation
      </th><th>TransE
      </th><td>0.145 ± 0.026</td>
      <td>0.061 ± 0.027</td>
      <td>0.164 ± 0.025</td>
      <td>0.301 ± 0.027</td>
   </tr>
   <tr>
      <th>DistMult
      </th><td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
   </tr>
   <tr>
      <th>ConvE
      </th><td><b>0.165 ± 0.017</b></td>
      <td><b>0.089 ± 0.015</b></td>
      <td><b>0.184 ± 0.022</b></td>
      <td><b>0.305 ± 0.020</b></td>
   </tr>
   
</tbody></table>
