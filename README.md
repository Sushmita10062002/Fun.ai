# Fun.ai
Fun.ai tries to complete the sentences in a humourous way given some input words.
I have finetuned GPT-2 model from huggingface library with a language model head on short jokes scrapped from reddit. I tested the model on some input and I got some good results. 
In this we simply tries to generate jokes, given the length of joke and number of jokes you want it to generate. 
Here we append 'JOKE:' at the start of every joke in our dataframe and '<|endoftext|>' at the end of each joke which tells our model that our joke has ended. At the time of inference

Here are some results:
![resutl1](https://github.com/Sushmita10062002/Fun.ai/blob/master/images/img1.png)
![result2](https://github.com/Sushmita10062002/Fun.ai/blob/master/images/img2.png)


## Dataset
You can access the jokes dataset [here](https://www.kaggle.com/datasets/abhinavmoudgil95/short-jokes). After forking the code, please add the data in the `inputs` folder if you want to train your own model.




