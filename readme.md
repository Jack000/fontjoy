# font vectors

>**Abstract: Font vectors are a form of transfer learning that can be used to compare visual features of fonts.**

>**Try the font pairing tool at http://fontjoy.com or play with the tensorboard projector at http://fontjoy.com/projector/ - I recommend the T-SNE visualization with a perplexity value of 30**


Extracting feature vectors from font images isn't an entirely [new idea](https://medium.com/ideo-stories/organizing-the-world-of-fonts-with-ai-7d9e49ff2b25).
You take some fonts, put them through a fixed feature extractor and get a representative vector on the other side. The basic idea behind this is covered in http://cs231n.github.io/transfer-learning/

The resulting vector is an *abstract representation* of what the font looks like. Because it's just a vector we can use vector arithmetic to compare different fonts.

You can actually do this with a couple of lines in keras:

```
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

```

For font comparison designers often use words like "Handgloves" which contain typographically distinguishing letters like e, a and n. I figured that a mnemonic word isn't necessary for a machine learning algo, so I used some mixed up letters:

![neural net input](http://fontjoy.com/github/input.png)

I treated each variant as a separate font, so the weights can be included in the font vector.

what can you use these vectors for? Well the simplest case is a visual similarity search:

![font similarity](http://fontjoy.com/github/similar.png)

if you've heard of word vectors, you've probably seen something like this:

![word2vec](http://fontjoy.com/github/word2vec.png)

it turns out you can do something similar with font vectors:

![font2vec](http://fontjoy.com/github/analogy1.png)

