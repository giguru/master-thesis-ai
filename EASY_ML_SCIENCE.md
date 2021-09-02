# Painless ML code publication checklist

In the field of science, there is little incentive to do some clean-up after the core of your work has been done.
However, you must be familiar with the time struggling to decypher someone's work when starting a new research
project. Let's make an effort together to make this process effortless! 

# Publication practises

1. Publish code

No excuses. Some code is always better then no code. No futher explanation required.

2. Dataset information

Data is key in ML. Datasets may float around on the internet in different versions, so preferably, *publish your
(processed) dataset AND the script to preprocess your dataset*. The latter is important, so people can see what
decisions you have made as a researcher.
Consequently, others can either use, alter or compare it. If you cannot publish the dataset, please publish some dataset
statistics.

3. Contribute to an open-source library!

Please consider taking a morning to publish your model/dataset to an open-source library which are easily accessible
like [HuggingFace](https://huggingface.co) or [ir_datasets](https://ir-datasets.com)
  
# Software environment

Please list the technology and used packages/libraries. For Python, simply run a `pip freeze > requirements.txt` and
publish it with your code.

# Feasibility

1. List your hardware

ML models/science may require huge amounts of resources. Not everyone has a million GPU's...
If you list your used hardware, others can quickly judge the feasibility to reproduce or continue on your work.

2. List your execution time.

How long does it take to preprocess data / train models / run execution times? A rough estimate is okay, since
everyone's hardware setup is different.

# Best coding practices
To summarise, the best coding practises assume others will follow up on your code.

1. Comment. Comment. Comment.

Code itself explain WHAT it's doing (adding, slicing, etc.), so commenting that is unnecessary, but it is useful to
comment WHY you are doing something,
e.g. no [magic numbers](https://en.wikipedia.org/wiki/Magic_number_(programming)#Unnamed_numerical_constants), explaining
if-else statements, explaining why you filter/slice, etc. By writing down the WHY, others may infer the implications of
changing your.

2. Small and isolated functions.

Please consider the fact that other researchers will want to use your work, because you're contributions are, of course,
_incredibly_ valuable! To make your work more accessible for others, please write small functions with descriptive names
and as isolated as possible instead of one endless script.

In addition, please be aware of global variables.