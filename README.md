# Real vs Recorded/artificial audio classifier

- [Dataset](/Dataset)
- [Audio Features](/Feature%20Extraction)
- [Models](/Models)
- [Results](/Results)

## Our Approach:

### Basic Pipeline
<img src="/approach.png" width="70%" />

### Novel Approach with Best Results
<img src="/gen%20Transformer%2BresSE%20net.png" width="70%" />

- This feature genuinization technique  learns a Transformer with CNN  the characteristics of only genuine speech. (Type of Pretraining)
- When the model takes spoof speech as input, it will generate very different output, that amplifies the difference to genuine speech.

<p float="middle">
  <img src="/Results/results.png" width="50%%" />
</p>

<p float="middle">
  <img src="/Results/LA%20Dev%20Set.png" width="35%" />
   
  <img src="/Results/LA%20Eval%20Set.png" width="35%" />
</p>

 
