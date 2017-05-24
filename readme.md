# font vectors

>**Abstract: Font vectors are a form of transfer learning that can be used to compare visual features of fonts.**

>**Try the font pairing tool at http://fontjoy.com or play with the tensorboard projector at http://fontjoy.com/projector/ - I recommend the T-SNE visualization with a perplexity value of 30**


![tensorboard visualization](http://fontjoy.com/github/screenshot.png)

Extracting feature vectors from images isn't an [entirely](http://blog.ethanrosenthal.com/2016/12/05/recasketch-keras/) new [idea](https://medium.com/ideo-stories/organizing-the-world-of-fonts-with-ai-7d9e49ff2b25).
You take some images, put them through a fixed feature extractor and get a representative vector on the other side. The basic idea behind this is covered in http://cs231n.github.io/transfer-learning/

If we use images of fonts, we get a vector that encodes the visual information of the font.

The font vector is an *abstract representation* of what the font looks like. Because it's just a vector we can use vector arithmetic to compare different fonts.

You can actually do this with a couple of lines in keras:

```
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

```

For font comparison designers often use words like "Handgloves" which contain typographically distinguishing letters like e, a and n. I figured that a mnemonic word isn't necessary for a machine learning algo, so I used mixed up letters:

![neural net input](http://fontjoy.com/github/input.png)

I treated each variant as a separate font, so the weights can be included in the font vector. There are 1883 different fonts in the dataset (from Google webfonts)

what can you use these vectors for? Well the simplest case is a visual similarity search:

![font similarity](http://fontjoy.com/github/similar.png)

if you've heard of word vectors, you've probably seen something like this:

![word2vec](http://fontjoy.com/github/word2vec.png)

it turns out you can do something similar with font vectors:

![font2vec](http://fontjoy.com/github/analogy1.png)

![font2vec](http://fontjoy.com/github/analogy2.png)

The results aren't always this clean but they usually make sense.

One of the more interesting problems in the design world is *font pairing*.

<img src="https://upload.wikimedia.org/wikipedia/commons/1/1d/Unclesamwantyou.jpg" alt="Uncle sam" style="width: 200px;"/>

Different fonts can be used to emphasize a message

![pairing examples](http://fontjoy.com/github/pairing.png)

Or to guide the eye and create visual interest.

The core process behind font pairing is somewhat paradoxical - we want fonts that contrast with each other, yet share certain similarities. How this is done in practice comes down to intuition, but we could try to narrow down the field with font vectors.

Here we reach a small issue - the vector measures that are commonly used for vector comparison don't convey this concept well:

- cosine distance (do the vectors point in the same direction?)

- Euclidean distance (are the vectors similar in direction and magnitude?)

- Hamming distance (are the vectors roughly similar?)

So we have to make up our own similarity measure:

![contrast distance](http://fontjoy.com/github/formula.png)

This is just the cosine distance split into two halves - the positive bits and the negative bits. By doing this our similarity measure will reward both similarities and dissimilarities - ie. we'll get fonts that are very similar in some respects but very different in other respects.

The contrasting pairings that are produced in this fashion don't always work together (clearly not all axis of contrast are visually pleasing), but the hit rate is surprisingly high.

![fontjoy demo](http://fontjoy.com/github/demo.png)

Another consideration for body text is legibility - many fonts that are suitable for titles aren't very readable at small sizes. To get the best balance between the 3 fonts we can try to optimize for best overall contrast, while weighing the legibility of the body font as a secondary factor.

Try the font paring generator at http://fontjoy.com