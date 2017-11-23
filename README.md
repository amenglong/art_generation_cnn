# Art Generation CNN
Art image generation using a pretrained VGG-19 deep convolutional neural network.

The program merges two images, a "content" image (C) and a "style" image (S), to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S.
In this example, we are going to generate an image of Barcelona city (content image C), mixed with a painting by Claude Monet, a leader of the impressionist movement (style image S):

<br/>
<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33163607-453dab62-d06a-11e7-803d-22be312a13eb.png" width="900"></p>

## Notes

### Trained model
<ul>
<li>Download trained model (510 MB) from <a href="https://onedrive.live.com/download?cid=B667AF4A4E4BA251&resid=B667AF4A4E4BA251%2141348&authkey=ANF7MA1OuYoB0c4">here</a>.</li>
<li>Save it in <b>/pretrained-model</b> folder</li>
</ul>

### Input images
For every new image:
<ul>
<li>Save it in /images folder</li>
<li>Size 400x300 (Width x Height) </li>
</ul>

### Generated image
Saved in <b>/output</b> folder
