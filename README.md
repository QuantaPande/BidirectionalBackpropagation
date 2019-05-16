# BidirectionalBackpropagation
An implementation of the bidirectional backpropagation algorithm outlined in the paper linked

Bidirectional backpropagation is a new way of training neural networks, which utilises the reverse network for increasing prediction strength with negligible increase in time complexity of training. 

Neural networks can be though of as set-level maps between the input space and the output space. We know that for a set level map, the inverse map always exists and maps from a set in the output space to a set in the input space. The paper ( http://sipi.usc.edu/~kosko/B-BP-SMC-Revised-13January2018.pdf ) outlines a way in which this inverse map can be used to increase the strength of the approximate map between the input and the output space.

The repository contains an implementation of bidirectional backpropagation using Tensorflow to simple MLP based architectures. It also contains an example code, with results of the inverse map for the MNIST Dataset.

## Demo images of the results of the reverse map on MNIST images:

These are blown-up results of the reverse map on the MNIST dataset
![result1789_Blow_Up](https://user-images.githubusercontent.com/28982129/57818908-e2e55680-773a-11e9-8a16-35076baf4f5b.png)
![result9328_Blow_Up](https://user-images.githubusercontent.com/28982129/57818945-07d9c980-773b-11e9-925d-1c120ce7571c.png)
![result2339_Blow_Up](https://user-images.githubusercontent.com/28982129/57818951-0b6d5080-773b-11e9-917a-4841e5d0e4eb.png)
![result978_Blow_Up](https://user-images.githubusercontent.com/28982129/57818953-0e684100-773b-11e9-994c-a98fac88a973.png)
![result2938_Blow_Up](https://user-images.githubusercontent.com/28982129/57818958-11633180-773b-11e9-86e9-4eb1fe8bf5f7.png)
![result1494_Blow_Up](https://user-images.githubusercontent.com/28982129/57818961-16c07c00-773b-11e9-90bc-77cc2d7355a1.png)
![result3341_Blow_Up](https://user-images.githubusercontent.com/28982129/57818968-1e802080-773b-11e9-8f77-b048713f52a9.png)
![result142_Blow_Up](https://user-images.githubusercontent.com/28982129/57818974-22ac3e00-773b-11e9-9877-4ad326449db0.png)
![result1150_Blow_Up](https://user-images.githubusercontent.com/28982129/57818975-25a72e80-773b-11e9-9119-26f1bf0566bd.png)
![result4557_Blow_Up](https://user-images.githubusercontent.com/28982129/57818979-2b047900-773b-11e9-9180-dec543e54fb8.png)

